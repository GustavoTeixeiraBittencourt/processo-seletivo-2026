"""Agente de recuperação (RAG).

Executa a busca vetorial para cada reformulação da query e mescla os resultados
por reciprocal-rank fusion — priorizando chunks que aparecem bem ranqueados
em múltiplas queries simultaneamente.

Design: roda `retrieve()` para a query original + cada reformulação (N+1 queries
no total), mescla por chunk_id mantendo o maior score de similaridade visto,
e ordena por esse score descendente. Isso maximiza recall sem sacrificar
precision, porque chunks irrelevantes que aparecem com score baixo em apenas
uma query ficam abaixo dos bem ranqueados.
"""

from __future__ import annotations

import time

from src.agents.state import PipelineState
from src.retrieval.retriever import retrieve


def _merge_hits(all_hits: list[list[dict]]) -> list[dict]:
    """Une resultados de múltiplas queries, mantendo o maior similarity por chunk."""
    best: dict[str, dict] = {}
    for hits in all_hits:
        for hit in hits:
            # Chave única por chunk (doc_id + chunk_index)
            meta = hit["metadata"]
            key = f"{meta['doc_id']}_{meta['chunk_index']}"
            if key not in best or hit["similarity"] > best[key]["similarity"]:
                best[key] = hit
    return sorted(best.values(), key=lambda h: h["similarity"], reverse=True)


def run(state: PipelineState) -> dict:
    start = time.monotonic()

    queries = [state["query_original"]] + state.get("query_reformulations", [])
    all_hits = [retrieve(q) for q in queries]
    merged = _merge_hits(all_hits)

    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "retrieved_chunks": merged,
        "retrieval_latency_ms": latency_ms,
    }
