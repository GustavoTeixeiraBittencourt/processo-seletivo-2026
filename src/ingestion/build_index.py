import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from .chunking import chunk_corpus
from .sources import extract_title, load_source_map

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
    source_map = load_source_map()

    title_cache: dict[str, str] = {}
    def doc_title(doc_id: str) -> str:
        if doc_id not in title_cache:
            text = (CLEAN_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
            title_cache[doc_id] = extract_title(text, fallback=doc_id)
        return title_cache[doc_id]

    metadatas = []
    for c in chunks:
        source_url = source_map.get(c.doc_id)
        if source_url is None:
            # falha alto em vez de gravar metadata incompleta em silêncio —
            # todo doc_id do corpus precisa estar listado em MANIFEST.md
            raise ValueError(
                f"doc_id={c.doc_id!r} sem source_url em {load_source_map.__module__}"
                " (MANIFEST.md desatualizado?)"
            )
        metadatas.append({
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "corpus_hash": chash,
            "embedding_model": EMBEDDING_MODEL,
            "source_url": source_url,
            "title": doc_title(c.doc_id),
        })

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
        metadatas = metadatas,
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