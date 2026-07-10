from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingestion.build_index import INDEX_DIR, EMBEDDING_MODEL, COLLECTION_NAME


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question: str, top_k: int = 5, min_similarity: float = 0.75):
    query_embedding = _get_model().encode([f"query: {question}"]).tolist()
    results = _get_collection().query(query_embeddings=query_embedding, n_results=top_k)
    documents = results["documents"] or [[]]
    metadatas = results["metadatas"] or [[]]
    distances = results["distances"] or [[]]

    hits = []
    for doc, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        similarity = 1 - distance  
        if similarity >= min_similarity:
            hits.append({"text": doc, "metadata": meta, "similarity": similarity})
    return hits
