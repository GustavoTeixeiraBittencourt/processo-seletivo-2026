from dataclasses import dataclass
from pathlib import Path
import tiktoken 

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 40
_enc = tiktoken.get_encoding("cl100k_base")

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str

def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_text = _enc.decode(tokens[start:end])
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_{index}",
            doc_id=doc_id, 
            chunk_index=index,
            text=chunk_text,
        ))
        if end == len(tokens):
            break
        
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
        index += 1
        
    return chunks 
    
def chunk_corpus(input_dir: Path) -> list[Chunk]:
    all_chunks = []
    for path in sorted(input_dir.rglob("*.md")):
        doc_id = path.relative_to(input_dir).with_suffix("").as_posix()
        all_chunks.extend(chunk_document(doc_id, path.read_text(encoding="utf-8")))
    return all_chunks