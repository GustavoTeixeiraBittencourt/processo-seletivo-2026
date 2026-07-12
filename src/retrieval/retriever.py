from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingestion.build_index import INDEX_DIR, EMBEDDING_MODEL, COLLECTION_NAME

''' Provisório até a fase final do projeto — calibrado a partir da distribuição medida em  data/processed/similarity_calibration.json (gerado por
 python -m src.retrieval._manual_check`): separa bem perguntas fora do
 domínio (~0.70-0.71) das in-corpus (~0.82-0.87), mas ainda deixa passar
 fallback "de fronteira" (mesmo domínio, fora do escopo do corpus,
 ~0.81-0.84) — essa faixa se sobrepõe ao range in-corpus e não é separável
 só por similaridade de cosseno; esses casos dependem do agente
 verificador, não do threshold.
'''
MIN_SIMILARITY_DEFAULT = 0.78


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question: str, top_k: int = 5, min_similarity: float = MIN_SIMILARITY_DEFAULT):
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
