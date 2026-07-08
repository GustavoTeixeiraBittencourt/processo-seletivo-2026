"""Cleaning pipeline for the raw LangGraph corpus (`corpus_langgraph/docs/`).

The corpus is a raw tarball snapshot of MkDocs-Material source, not rendered
HTML, so it still carries build-time artifacts that add semantic noise to
embeddings if left in: YAML front-matter, MkDocs admonition syntax
(`!!! tip`), unresolved `{% include-markdown %}` directives, and raw
HTML/JS/CSS blocks used for interactive widgets on the live site.

Run as a script to clean the whole corpus and print a report:

    python -m src.ingestion.cleaning
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import ftfy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "corpus_langgraph" / "docs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "clean"

# Below this word count, a "document" is almost certainly a broken/empty
# stub (e.g. an unresolved include) rather than real corpus content.
MIN_WORD_COUNT_WARNING = 50

FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)
INCLUDE_RE = re.compile(r'{%-?\s*include-markdown\s+"([^"]+)"\s*-?%}')
FIGURE_RE = re.compile(r"<figure[^>]*>(.*?)</figure>", re.DOTALL | re.IGNORECASE)
FIGCAPTION_RE = re.compile(r"<figcaption[^>]*>(.*?)</figcaption>", re.DOTALL | re.IGNORECASE)
BANG_ADMONITION_RE = re.compile(r'^(!{3}|\?{3})\s*([\w-]+)(?:\s+"([^"]*)")?\s*$')
TAB_ADMONITION_RE = re.compile(r'^={3}\s*"([^"]*)"\s*$')
LANG_OPEN_RE = re.compile(r"^\s*:::([\w-]+)\s*$")
LANG_CLOSE_RE = re.compile(r"^\s*:::\s*$")
AUTOREF_RE = re.compile(r"@\[([^\]]*)\]\[[^\]]*\]")
BLANK_RUN_RE = re.compile(r"\n{3,}")

# Raw HTML tags in this corpus are exclusively site-chrome/UI widgets
# (script blocks driving an interactive code snippet, layout <div>s with
# checkboxes, a branding <p>) — never prose worth keeping.
STRIPPED_TAGS = ("script", "style", "div", "p")

# The corpus pairs every Python example with an equivalent JS/TS one inside
# `:::python` / `:::js` fences. This project only ever runs Python, so the JS
# variant is duplicate content in a language the system doesn't use — keep
# just the Python fence's body and drop the JS one entirely rather than
# concatenating both unlabeled (which was silently doubling up near-identical
# text and injecting JS/TS syntax into a Python-only corpus).
KEEP_LANG_BLOCKS = {"python", "py"}
DROP_LANG_BLOCKS = {"js", "javascript", "ts", "typescript", "jsx", "tsx"}


@dataclass
class CleaningResult:
    doc_id: str
    source_path: Path
    text: str
    word_count: int
    warnings: list[str] = field(default_factory=list)


def _strip_balanced_tag(text: str, tag: str) -> str:
    """Remove every <tag ...>...</tag> block (tag and its content), tracking
    nesting depth so same-tag children (e.g. nested <div>s) don't confuse a
    naive first-close-tag match."""
    open_re = re.compile(rf"<{tag}(?:\s[^>]*)?>", re.IGNORECASE)
    close_re = re.compile(rf"</{tag}\s*>", re.IGNORECASE)
    out = []
    pos = 0
    depth = 0
    block_start = 0
    while True:
        if depth == 0:
            m = open_re.search(text, pos)
            if not m:
                out.append(text[pos:])
                break
            out.append(text[pos:m.start()])
            block_start = m.start()
            depth = 1
            pos = m.end()
        else:
            om = open_re.search(text, pos)
            cm = close_re.search(text, pos)
            if not cm:
                # Unbalanced markup: keep the rest untouched rather than
                # silently eating content we can't safely bound.
                out.append(text[block_start:])
                pos = len(text)
                break
            if om and om.start() < cm.start():
                depth += 1
                pos = om.end()
            else:
                depth -= 1
                pos = cm.end()
    return "".join(out)


def _replace_figures(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        caption = FIGCAPTION_RE.search(match.group(1))
        return caption.group(1).strip() if caption else ""

    return FIGURE_RE.sub(repl, text)


def _convert_admonitions(text: str) -> str:
    """Converts MkDocs `!!! tip "X"` / `??? example "X"` admonitions and
    `=== "X"` content-tabs into plain text: a bold label line followed by
    the dedented body. Both use the same "marker line, then a 4-space
    indented block" structure, and (unlike `:::lang` blocks) never nest in
    this corpus, so a flat single-level dedent is enough."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        bang = BANG_ADMONITION_RE.match(lines[i])
        tab = TAB_ADMONITION_RE.match(lines[i])
        if not bang and not tab:
            out.append(lines[i])
            i += 1
            continue
        if bang:
            kind, title = bang.group(2), bang.group(3)
            label = title or kind.replace("-", " ").title()
        else:
            label = tab.group(1)
        out.append(f"**{label}:**")
        i += 1
        body: list[str] = []
        while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
            body.append(lines[i][4:] if lines[i].startswith("    ") else lines[i])
            i += 1
        while body and body[-1].strip() == "":
            body.pop()
        out.extend(body)
        out.append("")
    return "\n".join(out)


def _process_lang_blocks(lines: list[str], warnings: list[str]) -> list[str]:
    """Resolves `:::python` / `:::js` language-variant fences with a small
    stack, pushing on open and popping on a bare `:::` close. These fences
    can nest (inside a `=== "Label"` tab body) and, in this corpus, the
    close marker's indentation doesn't reliably match the open marker's -
    so matching is by open/close order alone, not by indent."""
    stack: list[tuple[str, list[str]]] = []
    top: list[str] = []
    for line in lines:
        open_match = LANG_OPEN_RE.match(line)
        if open_match:
            stack.append((open_match.group(1).lower(), top))
            top = []
            continue
        if LANG_CLOSE_RE.match(line):
            if not stack:
                warnings.append("unmatched ':::' close marker with no open block, ignoring")
                continue
            lang, parent = stack.pop()
            if lang in KEEP_LANG_BLOCKS:
                parent.extend(top)
            elif lang not in DROP_LANG_BLOCKS:
                warnings.append(f"unknown ':::{lang}' block kept verbatim (not a recognized language)")
                parent.extend(top)
            top = parent
            continue
        top.append(line)
    if stack:
        warnings.append(
            f"{len(stack)} unclosed ':::' block(s) at end of document - keeping their content"
        )
        while stack:
            _, parent = stack.pop()
            parent.extend(top)
            top = parent
    return top


def _resolve_includes(
    text: str, file_path: Path, corpus_root: Path, warnings: list[str], depth: int = 0
) -> str:
    if depth > 3:
        warnings.append(f"include recursion too deep, giving up ({file_path.name})")
        return INCLUDE_RE.sub("", text)

    def repl(match: re.Match[str]) -> str:
        rel = match.group(1)
        target = (file_path.parent / rel).resolve()
        if target.is_file():
            included = ftfy.fix_text(target.read_text(encoding="utf-8"))
            return _resolve_includes(included, target, corpus_root, warnings, depth + 1)
        warnings.append(
            f"unresolved include '{rel}' (target not present in corpus snapshot)"
        )
        return ""

    return INCLUDE_RE.sub(repl, text)


def clean_text(raw_text: str, file_path: Path, corpus_root: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []

    text = ftfy.fix_text(raw_text)
    text = FRONTMATTER_RE.sub("", text)
    text = _resolve_includes(text, file_path, corpus_root, warnings)
    for tag in STRIPPED_TAGS:
        text = _strip_balanced_tag(text, tag)
    text = _replace_figures(text)
    text = "\n".join(_process_lang_blocks(text.split("\n"), warnings))
    text = _convert_admonitions(text)
    text = AUTOREF_RE.sub(r"\1", text)

    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RUN_RE.sub("\n\n", text)
    text = text.strip() + "\n"

    return text, warnings


def clean_file(path: Path, corpus_root: Path) -> CleaningResult:
    raw_text = path.read_text(encoding="utf-8")
    text, warnings = clean_text(raw_text, path, corpus_root)
    doc_id = path.relative_to(corpus_root).with_suffix("").as_posix()
    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT_WARNING:
        warnings.append(
            f"only {word_count} words after cleaning (< {MIN_WORD_COUNT_WARNING}) "
            "- likely a broken/near-empty document, review before keeping it in the corpus"
        )
    return CleaningResult(doc_id=doc_id, source_path=path, text=text, word_count=word_count, warnings=warnings)


def clean_corpus(input_dir: Path, output_dir: Path) -> list[CleaningResult]:
    paths = sorted(input_dir.rglob("*.md"))
    results = []
    for path in paths:
        result = clean_file(path, input_dir)
        out_path = output_dir / path.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.text, encoding="utf-8")
        results.append(result)
    return results


def _print_report(results: list[CleaningResult]) -> None:
    total_words = sum(r.word_count for r in results)
    print(f"Cleaned {len(results)} documents, {total_words} words total.\n")
    for r in results:
        flag = " !" if r.warnings else ""
        print(f"  {r.doc_id:<55} {r.word_count:>6} words{flag}")
        for w in r.warnings:
            print(f"      - {w}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON cleaning report (doc_id, word_count, warnings).",
    )
    args = parser.parse_args()

    results = clean_corpus(args.input_dir, args.output_dir)
    _print_report(results)

    if args.report:
        payload = [
            {"doc_id": r.doc_id, "word_count": r.word_count, "warnings": r.warnings}
            for r in results
        ]
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if any(r.warnings for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
