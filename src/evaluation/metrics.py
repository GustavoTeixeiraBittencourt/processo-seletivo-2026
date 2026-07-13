
from sentence_transformers import SentenceTransformer, util
from src.ingestion.build_index import EMBEDDING_MODEL

_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

def semantic_similarity(generated: str, expected: str) -> float:
    embeddings = _model.encode([f"query: {generated}", f"query: {expected}"])
    return float(util.cos_sim(embeddings[0], embeddings[1]))

