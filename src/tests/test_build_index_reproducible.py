import json
from pathlib import Path
from src.ingestion.build_index import  build_index, MANIFEST_PATH

def test_manifest_stable_across_rebuilds():
    build_index()
    first_manifest = json.loads(MANIFEST_PATH.read_text())
    build_index()
    second_manifest = json.loads(MANIFEST_PATH.read_text())
    assert first_manifest["corpus_hash"] == second_manifest["corpus_hash"]
    assert first_manifest["chunk_count"] == second_manifest["chunk_count"]