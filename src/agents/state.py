"""Estado compartilhado que flui por todos os nós do grafo LangGraph."""

from __future__ import annotations
from typing import TypedDict


class PipelineState(TypedDict):
    # Entrada
    query_original: str

    # Após reformulação
    query_reformulations: list[str]

    # Após recuperação (hits acima do threshold)
    retrieved_chunks: list[dict]

    # Decisão de fallback (gate inicial: dispara se retrieval não achou nada)
    fallback_triggered: bool
    fallback_reason: str | None

    # True se foi o verificador (não o fallback_decision inicial) que acionou
    # a busca web — caso "de fronteira": RAG achou chunks acima do threshold,
    # mas a resposta gerada não era fundamentada neles
    verifier_triggered_fallback: bool

    # Resultados da busca web (só populado se fallback)
    web_results: list[dict]

    # Contexto final enviado ao gerador (RAG ou web)
    context_used: list[dict]

    # Saída do gerador
    response_draft: str

    # Saída do verificador
    grounded: bool
    grounding_warnings: list[str]

    # Snapshot do primeiro passe (RAG) do gerador/verificador, preservado
    # para o trace mesmo quando o loop corretivo sobrescreve response_draft/
    # grounded com o resultado do segundo passe (via web)
    initial_response_draft: str
    initial_grounded: bool | None
    initial_grounding_warnings: list[str]

    # Resposta final ao usuário
    response_final: str

    # Latências por agente (em milissegundos)
    reformulation_latency_ms: int
    retrieval_latency_ms: int
    fallback_decision_latency_ms: int
    web_search_latency_ms: int
    generation_latency_ms: int
    verification_latency_ms: int
