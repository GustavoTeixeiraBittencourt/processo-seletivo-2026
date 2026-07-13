"""Agente de decisão de fallback.

Determina se o resultado do RAG é suficiente ou se é necessário acionar a
busca web. Esta decisão é DETERMINÍSTICA — não usa LLM, apenas regras
explícitas sobre os chunks recuperados.

Por que não usar LLM aqui: a decisão depende de um número (quantidade de
chunks acima do threshold), não de raciocínio semântico. LLM introduziria
latência e custo sem acrescentar qualidade à decisão.

Critério atual:
  - Nenhum chunk acima do threshold (retrieved_chunks vazio) → fallback
  - Pelo menos 1 chunk recuperado → segue com RAG (o verificador detecta
    alucinação se os chunks não forem suficientes para a pergunta)

Decisão de design documentada em data/processed/similarity_calibration.json:
perguntas "de fronteira" (mesmo domínio, versão fora do corpus) têm scores
sobrepostos ao range in-corpus (0.81-0.84 vs 0.82-0.87) — não são separáveis
por threshold. Esses casos dependem do verificador, não desta função.
"""

from __future__ import annotations

import time

from src.agents.state import PipelineState


def run(state: PipelineState) -> dict:
    start = time.monotonic()

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        triggered = True
        reason = (
            "Nenhum chunk recuperado acima do threshold de similaridade "
            f"(threshold={0.78}). O corpus (LangGraph v1.0.0) provavelmente "
            "não contém informação sobre esta pergunta."
        )
    else:
        triggered = False
        reason = None

    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "fallback_triggered": triggered,
        "fallback_reason": reason,
        "fallback_decision_latency_ms": latency_ms,
    }
