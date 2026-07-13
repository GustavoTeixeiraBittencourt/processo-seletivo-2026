"""Módulo de observabilidade — geração e persistência do trace JSON.

O trace é o "log de auditoria" de cada execução do pipeline. É obrigatório
pelo edital e permite ao avaliador entender o que o sistema fez internamente,
não só o que ele retornou.

Schema do trace: definido em docs/trace_schema.md e documentado aqui como
referência canônica.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACES_DIR = Path("traces")


def build_trace(state: dict, started_at: datetime) -> dict:
    """Constrói o dict do trace a partir do estado final do grafo."""

    # Latência total = soma das latências por agente (não wall-clock, para
    # evitar contabilizar tempo de I/O entre nós do grafo)
    latency_keys = [
        "reformulation_latency_ms",
        "retrieval_latency_ms",
        "fallback_decision_latency_ms",
        "web_search_latency_ms",
        "generation_latency_ms",
        "verification_latency_ms",
    ]
    total_latency_ms = sum(state.get(k, 0) for k in latency_keys)

    # Fontes únicas usadas (por doc_id ou URL)
    sources = list(
        {
            c.get("metadata", {}).get("doc_id", "")
            for c in state.get("context_used", [])
            if c.get("metadata")
        }
    )

    steps = [
        {
            "agent": "reformulator",
            "input": state.get("query_original", ""),
            "output": state.get("query_reformulations", []),
            "latency_ms": state.get("reformulation_latency_ms", 0),
        },
        {
            "agent": "retriever",
            "queries_used": [state.get("query_original", "")]
            + state.get("query_reformulations", []),
            "chunks_retrieved": [
                {
                    "chunk_id": (
                        f"{c['metadata']['doc_id']}_{c['metadata']['chunk_index']}"
                    ),
                    "similarity": round(c.get("similarity", 0), 4),
                    "title": c["metadata"].get("title", ""),
                    "source_url": c["metadata"].get("source_url", ""),
                }
                for c in state.get("retrieved_chunks", [])
            ],
            "fallback_triggered": state.get("fallback_triggered", False),
            "latency_ms": state.get("retrieval_latency_ms", 0),
        },
        {
            "agent": "fallback_decision",
            "triggered": state.get("fallback_triggered", False),
            "reason": state.get("fallback_reason"),
            "latency_ms": state.get("fallback_decision_latency_ms", 0),
        },
    ]

    if state.get("fallback_triggered"):
        steps.append(
            {
                "agent": "web_search",
                "query": state.get("query_original", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "score": r.get("score", 0),
                    }
                    for r in state.get("web_results", [])
                ],
                "latency_ms": state.get("web_search_latency_ms", 0),
            }
        )

    steps += [
        {
            "agent": "generator",
            "context_chunks_count": len(state.get("context_used", [])),
            "output_length_chars": len(state.get("response_draft", "")),
            "latency_ms": state.get("generation_latency_ms", 0),
        },
        {
            "agent": "verifier",
            "grounded": state.get("grounded", True),
            "warnings": state.get("grounding_warnings", []),
            "latency_ms": state.get("verification_latency_ms", 0),
        },
    ]

    return {
        "trace_id": str(uuid.uuid4()),
        "query_original": state.get("query_original", ""),
        "timestamp": started_at.isoformat(),
        "total_latency_ms": total_latency_ms,
        "fallback_used": state.get("fallback_triggered", False),
        "fallback_reason": state.get("fallback_reason"),
        "response_final": state.get("response_final", ""),
        "grounded": state.get("grounded", True),
        "grounding_warnings": state.get("grounding_warnings", []),
        "sources": sources,
        "steps": steps,
    }


def save_trace(trace: dict) -> Path:
    """Persiste o trace em /traces/ com nome baseado em timestamp + trace_id."""
    TRACES_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short_id = trace["trace_id"][:8]
    path = TRACES_DIR / f"trace_{ts}_{short_id}.json"
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
