"""Agente gerador de resposta.

Recebe o contexto (chunks do RAG ou resultados da busca web) e a pergunta
original, e usa o LLM para gerar uma resposta fundamentada nas fontes.

O prompt força citação explícita da fonte para dois fins:
1. Auditabilidade — o avaliador sabe de onde vem cada afirmação
2. Ancoragem — LLMs alucina menos quando obrigados a citar uma fonte real

Design da instrução "responda APENAS com base no contexto": reduz alucinação
ao custo de ocasionalmente produzir "não encontrei informação suficiente" para
perguntas borderline. Esse trade-off é aceitável — é melhor admitir
incerteza do que inventar dados técnicos sobre um framework.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._llm import call_llm
from src.agents.state import PipelineState

_SYSTEM_PROMPT = """\
Você é um assistente técnico especializado na documentação do LangGraph \
(framework Python de orquestração de agentes de IA).

Responda à pergunta do usuário usando APENAS as informações do contexto fornecido abaixo.

Regras obrigatórias:
- Cite a fonte de cada informação usada (título do documento ou URL)
- Se o contexto não contiver informação suficiente, diga explicitamente: \
  "Não encontrei informação suficiente no contexto disponível para responder com precisão."
- Não invente informações que não estejam no contexto
- Seja direto e técnico — o usuário é um desenvolvedor

Formato da resposta:
1. Resposta direta à pergunta
2. [Fonte: <título ou URL>]\
"""


def _build_context_block(context: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(context, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title") or meta.get("doc_id", f"Documento {i}")
        source = meta.get("source_url", "")
        text = chunk.get("text", "")
        parts.append(f"--- Trecho {i} ({title}) ---\n{text}\nFonte: {source}")
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    start = time.monotonic()

    # Se o contexto ainda não foi definido (caminho RAG sem fallback),
    # usa os chunks recuperados pelo agente de retrieval
    context = state.get("context_used") or state.get("retrieved_chunks", [])

    context_block = _build_context_block(context)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Contexto:\n{context_block}\n\nPergunta: {state['query_original']}"
        ),
    ]

    response = call_llm(messages)
    draft = response.content.strip()

    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "context_used": context,
        "response_draft": draft,
        "generation_latency_ms": latency_ms,
    }
