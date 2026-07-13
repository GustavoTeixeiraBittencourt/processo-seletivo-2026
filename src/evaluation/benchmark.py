"""Benchmark do pipeline RAG contra os pares de qa_pairs.json.

Uso padrão (split=validation, 8 perguntas):
    python -m src.evaluation.benchmark

Para rodar contra o split de test (12 perguntas):
    python -m src.evaluation.benchmark --split test

Para rodar todos os 20 pares:
    python -m src.evaluation.benchmark --split all
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.metrics import semantic_similarity
from src.orchestration.graph import run_pipeline

QA_PATH = Path("benchmark/qa_pairs.json")
RESULTS_DIR = Path("benchmark")


def load_pairs(split: str) -> list[dict]:
    pairs = json.loads(QA_PATH.read_text(encoding="utf-8"))
    if split == "all":
        return pairs
    return [p for p in pairs if p["split"] == split]


def _evaluate_pair(pair: dict) -> dict:
    pair_id = pair["id"]
    question = pair["question"]
    is_fallback_expected: bool = pair["is_fallback"]
    expected_answer: str | None = pair.get("expected_answer")
    expected_doc: str | None = pair.get("expected_source_doc_id")

    try:
        response_final, trace, trace_path = run_pipeline(question)

        fallback_triggered: bool = trace["fallback_used"]
        fallback_correct: bool = fallback_triggered == is_fallback_expected

        # Similaridade semântica — só calculada quando há resposta esperada (RAG)
        similarity: float | None = None
        if expected_answer is not None:
            similarity = round(semantic_similarity(response_final, expected_answer), 4)

        # Recall do documento — só calculado quando há doc esperado (RAG)
        doc_recall: bool | None = None
        if expected_doc is not None:
            doc_recall = expected_doc in trace.get("sources", [])

        return {
            "id": pair_id,
            "split": pair["split"],
            "is_fallback_expected": is_fallback_expected,
            "fallback_triggered": fallback_triggered,
            "fallback_correct": fallback_correct,
            "semantic_similarity": similarity,
            "doc_recall": doc_recall,
            "grounded": trace["grounded"],
            "grounding_warnings": trace["grounding_warnings"],
            "total_latency_ms": trace["total_latency_ms"],
            "trace_id": trace["trace_id"],
            "trace_path": str(trace_path),
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "id": pair_id,
            "split": pair["split"],
            "is_fallback_expected": is_fallback_expected,
            "fallback_triggered": None,
            "fallback_correct": None,
            "semantic_similarity": None,
            "doc_recall": None,
            "grounded": None,
            "grounding_warnings": [],
            "total_latency_ms": None,
            "trace_id": None,
            "trace_path": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _aggregate(results: list[dict]) -> dict:
    valid = [r for r in results if r["error"] is None]

    sims = [r["semantic_similarity"] for r in valid if r["semantic_similarity"] is not None]
    recalls = [r["doc_recall"] for r in valid if r["doc_recall"] is not None]
    fallback_checks = [r["fallback_correct"] for r in valid if r["fallback_correct"] is not None]
    latencies = [r["total_latency_ms"] for r in valid if r["total_latency_ms"] is not None]
    grounded_flags = [r["grounded"] for r in valid if r["grounded"] is not None]

    def avg(lst: list) -> float | None:
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "total_pairs": len(results),
        "errors": len(results) - len(valid),
        "avg_semantic_similarity": avg(sims),
        "fallback_classification_accuracy": avg(fallback_checks),
        "doc_recall_rate": avg(recalls),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "grounded_rate": avg(grounded_flags),
    }


def run_benchmark(split: str = "validation") -> dict:
    pairs = load_pairs(split)
    total = len(pairs)
    print(f"\nBenchmark iniciado — split={split!r}, {total} pares\n", flush=True)

    results: list[dict] = []
    for i, pair in enumerate(pairs, 1):
        prefix = f"[{i}/{total}] {pair['id']}"
        snippet = pair["question"][:55] + "..."
        print(f"  {prefix}: {snippet}", flush=True)

        result = _evaluate_pair(pair)
        results.append(result)

        if result["error"]:
            print(f"         ERRO: {result['error']}", flush=True)
        else:
            sim_part = (
                f"sim={result['semantic_similarity']:.3f}"
                if result["semantic_similarity"] is not None
                else "sim=N/A (fallback)"
            )
            fb_part = (
                "fallback=OK" if result["fallback_correct"] else "fallback=ERRADO"
            )
            doc_part = (
                f"recall={'OK' if result['doc_recall'] else 'MISS'}"
                if result["doc_recall"] is not None
                else "recall=N/A"
            )
            print(
                f"         {sim_part} | {fb_part} | {doc_part}"
                f" | {result['total_latency_ms']}ms",
                flush=True,
            )

    summary = _aggregate(results)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "summary": summary,
        "results": results,
    }

    out_path = RESULTS_DIR / f"results_{ts}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'─'*55}", flush=True)
    print(f"Resultados salvos em: {out_path}", flush=True)
    print(f"{'─'*55}", flush=True)
    print(f"  Similaridade semântica média : {summary['avg_semantic_similarity']}", flush=True)
    print(f"  Acerto classificação fallback: {summary['fallback_classification_accuracy']}", flush=True)
    print(f"  Doc recall rate              : {summary['doc_recall_rate']}", flush=True)
    print(f"  Latência média               : {summary['avg_latency_ms']} ms", flush=True)
    print(f"  Taxa de fundamentação        : {summary['grounded_rate']}", flush=True)
    print(f"  Pares com erro               : {summary['errors']}/{summary['total_pairs']}", flush=True)
    print(f"{'─'*55}\n", flush=True)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark do pipeline RAG")
    parser.add_argument(
        "--split",
        choices=["validation", "test", "all"],
        default="validation",
        help="Split a avaliar (default: validation)",
    )
    args = parser.parse_args()
    run_benchmark(split=args.split)
