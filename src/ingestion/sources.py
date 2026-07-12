import re
from pathlib import Path

MANIFEST_PATH = Path("corpus_langgraph/MANIFEST.md")
_LINE_RE = re.compile(r"^-\s*`(docs/[^`]+\.md)`\s*—\s*(\S+)")


def load_source_map(manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    """doc_id -> raw source URL, parsed from MANIFEST.md (single source of truth,
    avoids duplicating the URL list and letting it drift out of sync)."""
    mapping: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = _LINE_RE.match(line.strip())
        if not match:
            continue
        rel_path, url = match.groups()
        doc_id = Path(rel_path).relative_to("docs").with_suffix("").as_posix()
        mapping[doc_id] = url
    return mapping


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback
