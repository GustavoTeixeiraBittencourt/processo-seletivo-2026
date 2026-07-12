from rank_bm25 import BM25Okapi

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