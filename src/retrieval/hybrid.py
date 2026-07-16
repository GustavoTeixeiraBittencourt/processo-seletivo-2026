"""Recuperação híbrida: BM25 léxico + Reciprocal Rank Fusion (RRF).

Integrado em `src/agents/retriever.py`, que usa `bm25_ranking()` para
re-ranquear os candidatos densos já filtrados pelo threshold de similaridade
(`src/retrieval/retriever.py`). BM25 aqui não substitui nem amplia o gate de
threshold — só reordena/reforça o que a busca densa já qualificou, cobrindo
o caso em que uma pergunta usa termos exatos (nomes de função, classes) que
o embedding denso rankeia mais abaixo do que deveria. Ver
`docs/decisoes_tecnicas.md` para a justificativa completa dessa escolha de
design (por que o threshold denso continua sendo o único gate de fallback).
"""

from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from src.ingestion.build_index import COLLECTION_NAME, INDEX_DIR


def build_bm25(chunks: list[str]) -> BM25Okapi:
    tokenized = [c.lower().split(" ") for c in chunks]
    return BM25Okapi(tokenized)


def reciprocal_rank_fusion(bm25_ranking: list[str], dense_ranking: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(bm25_ranking):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank + 1)
    for rank, chunk_id in enumerate(dense_ranking):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank + 1)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:k]


@lru_cache(maxsize=1)
def _get_corpus_bm25_index() -> tuple[BM25Okapi, tuple[str, ...]]:
    """Carrega todos os chunks do Chroma e monta o índice BM25 em memória.

    Memoizado com lru_cache (mesmo padrão de `src/retrieval/retriever.py`)
    porque o corpus não muda dentro do processo — reconstruir o índice BM25
    a cada query seria refazer trabalho idêntico.
    """
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents"])
    return build_bm25(data["documents"]), tuple(data["ids"])


def bm25_ranking(question: str, top_k: int = 10) -> list[str]:
    """Retorna os `top_k` chunk_ids mais relevantes por score BM25 (léxico).

    Sem filtro de score mínimo — diferente da busca densa, o score do BM25
    não é comparável entre queries nem calibrado contra um threshold; quem
    decide o que entra na resposta final é o gate de similaridade densa em
    `src/agents/retriever.py`, não este ranking isolado.
    """
    bm25, ids = _get_corpus_bm25_index()
    scores = bm25.get_scores(question.lower().split(" "))
    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_k] if score > 0]