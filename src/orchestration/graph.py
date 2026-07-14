"""Grafo LangGraph — orquestração do pipeline multiagente.

Define o fluxo de execução conectando os 6 agentes em um StateGraph.
O estado (PipelineState) flui pelos nós; cada nó lê o que precisa e
escreve apenas seus próprios campos — sem acoplamento direto entre agentes.

Fluxo:
  START → reformulator → retriever → fallback_decision
                                         ↓
                               [condicional: rag ou web?]
                              /                          \
                       [rag path]                  [web path]
                           ↓                            ↓
                       generator  ←──────────  web_search
                           ↓
                       verifier
                           ↓
                  [condicional: fundamentado?]
                  /                          \
              [sim/já tentou web]      [não E ainda não tentou web]
                  ↓                            ↓
                 END                      web_search (retry)
                                                ↓
                                           generator (2º passe)
                                                ↓
                                           verifier (2º passe) → END

O loop de fallback corretivo existe porque o threshold de similaridade
sozinho não separa perguntas "de fronteira" (mesmo domínio, fora do escopo
do corpus — ver data/processed/similarity_calibration.json) de perguntas
genuinamente respondíveis pelo corpus: ambas retornam chunks acima do
threshold. fallback_decision (gate inicial) só pega o caso "zero chunks";
o caso de fronteira só é detectável depois, pelo verifier, checando se a
resposta gerada realmente se sustenta no que foi recuperado. Sem esse loop,
uma resposta que o próprio verifier marca como não fundamentada ainda assim
seria entregue ao usuário como resposta final — o verifier virava só um
log, não uma ação corretiva. A checagem `not state["fallback_triggered"]`
no roteador garante no máximo uma tentativa de correção por execução (sem
isso, um segundo veredito "não fundamentado" via web faria o grafo ciclar
indefinidamente).
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from src.agents import (
    fallback_decision,
    generator,
    reformulator,
    retriever,
    verifier,
    web_search,
)
from src.agents.state import PipelineState
from src.observability.tracer import build_trace, save_trace

# Estado inicial padrão — garante que todos os campos TypedDict existem
# desde o início, evitando KeyError em nós que leem campos ainda não populados
_INITIAL_STATE: PipelineState = {
    "query_original": "",
    "query_reformulations": [],
    "retrieved_chunks": [],
    "fallback_triggered": False,
    "fallback_reason": None,
    "verifier_triggered_fallback": False,
    "web_results": [],
    "context_used": [],
    "response_draft": "",
    "grounded": True,
    "grounding_warnings": [],
    "response_final": "",
    "initial_response_draft": "",
    "initial_grounded": None,
    "initial_grounding_warnings": [],
    "reformulation_latency_ms": 0,
    "retrieval_latency_ms": 0,
    "fallback_decision_latency_ms": 0,
    "web_search_latency_ms": 0,
    "generation_latency_ms": 0,
    "verification_latency_ms": 0,
}


def _route_after_fallback_decision(state: PipelineState) -> str:
    """Aresta condicional: decide se vai para web_search ou direto para generator."""
    return "web_search" if state["fallback_triggered"] else "generator"


def _route_after_verifier(state: PipelineState) -> str:
    """Aresta condicional: fallback corretivo se a resposta não é fundamentada
    e a busca web ainda não foi tentada nesta execução (ver docstring do módulo)."""
    if not state["grounded"] and not state["fallback_triggered"]:
        return "web_search"
    return "end"


def _build() -> object:
    graph = StateGraph(PipelineState)

    graph.add_node("reformulator", reformulator.run)
    graph.add_node("retriever", retriever.run)
    graph.add_node("fallback_decision", fallback_decision.run)
    graph.add_node("web_search", web_search.run)
    graph.add_node("generator", generator.run)
    graph.add_node("verifier", verifier.run)

    graph.add_edge(START, "reformulator")
    graph.add_edge("reformulator", "retriever")
    graph.add_edge("retriever", "fallback_decision")
    graph.add_conditional_edges(
        "fallback_decision",
        _route_after_fallback_decision,
        {"web_search": "web_search", "generator": "generator"},
    )
    graph.add_edge("web_search", "generator")
    graph.add_edge("generator", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"web_search": "web_search", "end": END},
    )

    return graph.compile()


# Instância compilada — criada uma vez, reutilizada em todas as chamadas
_pipeline = _build()


def run_pipeline(question: str) -> tuple[str, dict, object]:
    """Executa o pipeline completo para uma pergunta.

    Returns:
        (resposta_final, trace_dict, caminho_do_trace_salvo)
    """
    started_at = datetime.now(timezone.utc)

    initial = dict(_INITIAL_STATE)
    initial["query_original"] = question

    final_state = _pipeline.invoke(initial)

    trace = build_trace(final_state, started_at)
    trace_path = save_trace(trace)

    return final_state["response_final"], trace, trace_path
