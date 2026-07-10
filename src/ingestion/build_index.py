import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from .chunking import chunk_corpus

CLEAN_DIR = Path("data/processed/clean")
INDEX_DIR = Path("data/processed/chroma_index")
MANIFEST_PATH = Path("data/processed/index_manifest.json")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "langgraph_docs"

def corpus_hash(clean_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(clean_dir.rglob("*.md")):
        h.update(path.read_text(encoding="utf-8").encode("utf-8"))
    return h.hexdigest()

def build_index() -> None:
    chash = corpus_hash(CLEAN_DIR)

    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text())
        if old.get("corpus_hash") == chash:
            print("corpus_hash inalterado... pulando rebuild")
            return
    chunks = chunk_corpus(CLEAN_DIR)
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode([f"passage: {c.text}" for c in chunks]).tolist()
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids = [c.chunk_id for c in chunks],
        documents = [c.text for c in chunks],
        embeddings = embeddings,
        metadatas=[{"doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "corpus_hash": chash,
                    "embedding_model": EMBEDDING_MODEL,} for c in chunks]
    )

    MANIFEST_PATH.write_text(json.dumps({
        "corpus_hash": chash,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_count": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"Indice Construído com {len(chunks)} chunks, hash {chash[:12]} e salvo em {INDEX_DIR}.")


if __name__ == "__main__":
    build_index()