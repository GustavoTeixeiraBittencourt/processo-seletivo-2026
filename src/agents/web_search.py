"""Agente de busca web (fallback).

Acionado em dois cenários (ver orchestration/graph.py):
1. Fallback inicial: fallback_decision não achou nenhum chunk acima do
   threshold — vai direto para cá, antes do generator.
2. Fallback corretivo: o RAG achou chunks (caso "de fronteira", threshold
   passou mas a pergunta está fora do escopo real do corpus — ver
   data/processed/similarity_calibration.json) e o verifier marcou a
   resposta como não fundamentada. Nesse caso o grafo volta para cá depois
   do verifier, e o generator roda de novo com contexto da web.

Usa o Tavily — uma API de busca feita especificamente para sistemas RAG:
retorna trechos já limpos e relevantes ao invés de HTML bruto, eliminando a
necessidade de scraping e limpeza pós-busca.

Por que Tavily e não a API do Google/Bing: Tavily inclui um score de
relevância por resultado e extrai automaticamente o conteúdo textual das
páginas, o que se encaixa diretamente no formato esperado pelo gerador.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from tavily import TavilyClient

from src.agents.state import PipelineState

load_dotenv()

_MAX_RESULTS = 5


def _get_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY não encontrada. "
            "Adicione ao .env com base em .env.example."
        )
    return TavilyClient(api_key=api_key)


def run(state: PipelineState) -> dict:
    start = time.monotonic()
    client = _get_client()

    # Se chegamos aqui com fallback_triggered ainda False, foi o verifier que
    # nos mandou de volta (caso 2 do docstring acima) — não o fallback_decision
    # inicial (caso 1, onde fallback_triggered já veio True)
    is_corrective = not state.get("fallback_triggered", False)

    # Usa a query original para busca web — reformulações são otimizadas para
    # embeddings, não para motores de busca tradicionais
    query = state["query_original"]
    response = client.search(query, max_results=_MAX_RESULTS)

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": round(r.get("score", 0.0), 4),
        }
        for r in response.get("results", [])
    ]

    # Normaliza para o mesmo formato de context_used que o retriever produz,
    # facilitando o reuso do generator independente da origem dos dados
    context_used = [
        {
            "text": r["content"],
            "metadata": {
                "doc_id": r["url"],
                "title": r["title"],
                "source_url": r["url"],
                "chunk_index": i,
            },
            "similarity": r["score"],
        }
        for i, r in enumerate(results)
    ]

    latency_ms = int((time.monotonic() - start) * 1000)
    update = {
        "web_results": results,
        "context_used": context_used,
        # Acumula em vez de sobrescrever: no fallback corretivo, esta é a
        # segunda chamada de web_search dentro da mesma execução
        "web_search_latency_ms": state.get("web_search_latency_ms", 0) + latency_ms,
        "fallback_triggered": True,
    }
    if is_corrective:
        update["fallback_reason"] = (
            "Resposta gerada a partir dos chunks recuperados não foi "
            "considerada fundamentada pelo verificador (caso 'de fronteira': "
            "similaridade passou do threshold, mas o conteúdo não sustenta a "
            "pergunta) — acionando busca web como fallback corretivo."
        )
        update["verifier_triggered_fallback"] = True
    return update
