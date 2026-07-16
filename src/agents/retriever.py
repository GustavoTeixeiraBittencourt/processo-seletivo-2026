"""Agente de recuperação (RAG) — busca densa + híbrida (BM25 + RRF).

Executa a busca vetorial para cada reformulação da query e re-ranqueia os
candidatos com BM25 via Reciprocal Rank Fusion, priorizando chunks que
aparecem bem ranqueados tanto na busca densa quanto na léxica.

Design: roda `retrieve()` (denso, já filtrado pelo threshold de similaridade
em `src/retrieval/retriever.py`) para a query original + cada reformulação.
Chunks que nunca aparecem na busca densa filtrada — para nenhuma das
queries — nunca entram no resultado, mesmo que o BM25 os ranqueie bem: o
threshold denso continua sendo o único gate de "o corpus tem ou não tem
resposta" (usado por `fallback_decision.py`). BM25 entra só depois desse
gate, para reordenar os candidatos já qualificados — cobre o caso de
perguntas com termos exatos (nomes de função/classe) que a busca densa
rankeia mais abaixo do que deveria, sem arriscar admitir chunk irrelevante
que só bateu por sobreposição lexical (ex.: uma palavra comum repetida no
corpus). Ver `docs/decisoes_tecnicas.md` para a justificativa completa.
"""

from __future__ import annotations

import time

from src.agents.state import PipelineState
from src.retrieval.hybrid import bm25_ranking, reciprocal_rank_fusion
from src.retrieval.retriever import retrieve

_DENSE_CANDIDATES_PER_QUERY = 10


def _chunk_key(hit: dict) -> str:
    meta = hit["metadata"]
    return f"{meta['doc_id']}_{meta['chunk_index']}"


def run(state: PipelineState) -> dict:
    start = time.monotonic()

    queries = [state["query_original"]] + state.get("query_reformulations", [])

    hits_by_id: dict[str, dict] = {}
    bm25_hit_ids: set[str] = set()
    per_query_fused: list[list[str]] = []

    for query in queries:
        dense_hits = retrieve(query, top_k=_DENSE_CANDIDATES_PER_QUERY)
        dense_ranking = []
        for hit in dense_hits:
            key = _chunk_key(hit)
            dense_ranking.append(key)
            if key not in hits_by_id or hit["similarity"] > hits_by_id[key]["similarity"]:
                hits_by_id[key] = hit

        if not dense_ranking:
            # Nenhum candidato passou no threshold denso para esta query —
            # BM25 não tem o que reordenar aqui (ver docstring do módulo).
            continue

        bm25_ids = bm25_ranking(query, top_k=_DENSE_CANDIDATES_PER_QUERY)
        bm25_hit_ids.update(bm25_ids)
        per_query_fused.append(
            reciprocal_rank_fusion(bm25_ids, dense_ranking, k=len(dense_ranking))
        )

    # Combina o ranking fundido (denso+BM25) de cada query num ranking único
    # — mesma lógica de RRF, agora entre queries em vez de entre métodos.
    combined_scores: dict[str, float] = {}
    for ranking in per_query_fused:
        for rank, chunk_id in enumerate(ranking):
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + 1 / (rank + 1)

    ordered_ids = sorted(combined_scores, key=lambda cid: combined_scores[cid], reverse=True)

    merged = []
    for chunk_id in ordered_ids:
        # Filtro de segurança: só inclui chunk que de fato qualificou pelo
        # threshold denso em alguma query — preserva o comportamento de
        # fallback mesmo com o re-ranking híbrido.
        if chunk_id not in hits_by_id:
            continue
        hit = dict(hits_by_id[chunk_id])
        hit["matched_by"] = "hybrid" if chunk_id in bm25_hit_ids else "dense"
        merged.append(hit)

    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "retrieved_chunks": merged[:10],
        "retrieval_latency_ms": latency_ms,
    }
