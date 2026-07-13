"""Agente reformulador de query.

Recebe a pergunta original e usa o LLM para gerar N reformulações/expansões.
O objetivo é aumentar o recall na busca vetorial: formulações diferentes
ativam chunks diferentes no espaço de embeddings, cobrir mais do corpus.

Exemplo:
  entrada: "como criar um agente?"
  saída:   ["como implementar um agente LangGraph em Python?",
             "tutorial criação de agente com StateGraph"]
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._llm import call_llm
from src.agents.state import PipelineState

_N_REFORMULATIONS = 2

_SYSTEM_PROMPT = """\
Você é um especialista em recuperação de informação para sistemas RAG \
sobre a documentação do LangGraph (framework Python de orquestração de agentes de IA).

Dada uma pergunta do usuário, gere exatamente {n} reformulações alternativas que:
- Preservem o significado original
- Usem vocabulário técnico diferente (sinônimos, abreviações, termos do domínio)
- Cubram diferentes ângulos semânticos da mesma dúvida

Responda SOMENTE com as reformulações numeradas (1. ... 2. ...), sem preâmbulo.\
"""


def _parse_reformulations(raw: str, n: int) -> list[str]:
    results = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove prefixos "1. ", "1) ", "1- " etc.
        if line and line[0].isdigit():
            rest = line[1:].lstrip(".)-: ")
            if rest:
                results.append(rest)
            continue
        results.append(line)
    return results[:n]


def run(state: PipelineState) -> dict:
    start = time.monotonic()

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT.format(n=_N_REFORMULATIONS)),
        HumanMessage(
            content=f"Pergunta original: {state['query_original']}\n\n"
            f"Gere {_N_REFORMULATIONS} reformulações:"
        ),
    ]

    response = call_llm(messages)
    reformulations = _parse_reformulations(response.content, _N_REFORMULATIONS)

    # Garante que nunca retornamos lista vazia — usa a original como fallback
    if not reformulations:
        reformulations = [state["query_original"]]

    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "query_reformulations": reformulations,
        "reformulation_latency_ms": latency_ms,
    }
