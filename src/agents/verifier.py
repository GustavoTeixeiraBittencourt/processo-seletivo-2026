"""Agente verificador (grounding check).

Compara a resposta gerada pelo agente generator com o contexto-fonte e
determina se cada afirmação da resposta tem suporte no contexto.

Por que é necessário: o generator pode produzir respostas plausíveis mas
não fundamentadas, especialmente em perguntas "de fronteira" (mesmo domínio
mas fora do corpus) que passaram pelo threshold do retriever. O verificador
é a segunda linha de defesa contra alucinação.

Formato de saída do LLM: texto estruturado com linha "VEREDICTO: FUNDAMENTADO"
ou "VEREDICTO: NÃO FUNDAMENTADO" seguida de "AVISOS: ..." — mais robusto que
JSON parsing para respostas curtas, sem dependência de um esquema Pydantic.

Caso especial — recusa explícita: quando o generator não acha suporte
suficiente, ele é instruído (generator.py) a responder com uma frase fixa
("Não encontrei informação suficiente..."). Resolvido por checagem
determinística (_is_explicit_refusal), não perguntando ao LLM: mais rápido,
mais barato, e sem o risco (observado em produção, ver
docs/decisoes_tecnicas.md) de o verificador penalizar a própria honestidade
que ele foi desenhado para reconhecer. Mas a recusa só é *trivialmente*
fundamentada se a busca web já foi tentada (fallback_triggered=True) — se
for a primeira recusa, ainda só com contexto do RAG, ela precisa continuar
contando como "não fundamentado" para o roteador (_route_after_verifier em
orchestration/graph.py) mandar o grafo pelo loop corretivo antes de desistir.
Sem essa distinção, uma recusa honesta no primeiro passe encerraria o grafo
sem nunca tentar a web — o mesmo problema que o loop corretivo foi criado
para resolver, só que por um caminho diferente.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._llm import call_llm
from src.agents.state import PipelineState

# Precisa bater com a frase exata instruída em generator.py — ver docstring
# do módulo, seção "Caso especial"
_REFUSAL_MARKER = "não encontrei informação suficiente"

_SYSTEM_PROMPT = """\
Você é um verificador de qualidade para sistemas RAG. Sua tarefa é checar se \
uma resposta gerada está fundamentada no contexto-fonte fornecido.

Instruções:
1. Leia o contexto e a resposta gerada
2. Identifique afirmações na resposta que NÃO têm suporte no contexto
3. Responda EXATAMENTE neste formato (sem texto adicional):

VEREDICTO: FUNDAMENTADO
AVISOS: nenhum

ou

VEREDICTO: NÃO FUNDAMENTADO
AVISOS: <lista de afirmações sem suporte no contexto, uma por linha>\
"""


def _is_explicit_refusal(draft: str) -> bool:
    return _REFUSAL_MARKER in draft.lower()


def _parse_verdict(raw: str) -> tuple[bool, list[str]]:
    """Extrai (grounded, warnings) do texto de resposta do LLM."""
    grounded = True
    warnings: list[str] = []

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("VEREDICTO:"):
            verdict_text = line.split(":", 1)[1].strip().upper()
            grounded = "NÃO" not in verdict_text and "NAO" not in verdict_text
        elif line.startswith("AVISOS:"):
            aviso = line.split(":", 1)[1].strip()
            if aviso and aviso.lower() != "nenhum":
                warnings.append(aviso)
        elif warnings or (not grounded and line):
            # Linhas adicionais de avisos (quando há múltiplos)
            warnings.append(line)

    return grounded, warnings


def _truncate_context(context: list[dict], max_chars: int = 6000, max_chunk_chars: int = 1500) -> str:
    """Trunca o contexto para evitar exceder a janela de tokens da Groq.

    Corta em dois níveis — por trecho e no total — porque conteúdo vindo da
    busca web (web_search.py) não tem limite de tamanho por si só (ao
    contrário dos chunks do RAG, já truncados na indexação). Sem o limite
    por trecho, um único resultado web longo o suficiente estouraria
    max_chars sozinho e seria descartado por inteiro pelo corte de total,
    deixando o verificador rodar contra contexto vazio sem nenhum aviso.
    """
    parts = []
    total = 0
    for chunk in context:
        text = chunk.get("text", "")[:max_chunk_chars]
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def run(state: PipelineState) -> dict:
    start = time.monotonic()

    context = state.get("context_used", [])
    draft = state.get("response_draft", "")

    # Se não há draft (pipeline falhou antes), assume não fundamentado
    if not draft:
        latency_ms = int((time.monotonic() - start) * 1000)
        return {
            "grounded": False,
            "grounding_warnings": ["Nenhuma resposta foi gerada pelo agente anterior."],
            "response_final": "",
            "verification_latency_ms": latency_ms,
        }

    if _is_explicit_refusal(draft):
        # Só é trivialmente fundamentada se a web já foi tentada — ver
        # docstring do módulo ("Caso especial — recusa explícita")
        already_tried_web = state.get("fallback_triggered", False)
        grounded = already_tried_web
        warnings = (
            []
            if already_tried_web
            else ["Resposta é uma recusa explícita — RAG não encontrou suporte suficiente."]
        )

        latency_ms = int((time.monotonic() - start) * 1000)
        update = {
            "grounded": grounded,
            "grounding_warnings": warnings,
            "response_final": draft,
            "verification_latency_ms": latency_ms,
        }
        if not state.get("initial_response_draft"):
            update["initial_response_draft"] = draft
            update["initial_grounded"] = grounded
            update["initial_grounding_warnings"] = warnings
        return update

    context_text = _truncate_context(context)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Contexto-fonte:\n{context_text}\n\n"
                f"Resposta gerada:\n{draft}"
            )
        ),
    ]

    response = call_llm(messages)
    grounded, warnings = _parse_verdict(response.content)

    # A resposta final é sempre o draft mais recente — se grounded=False e o
    # grafo ainda não tentou fallback (ver _route_after_verifier em
    # orchestration/graph.py), o pipeline volta por web_search e roda
    # generator/verifier de novo; response_final é sobrescrita nesse segundo
    # passe. initial_* preserva o veredito do primeiro passe (só é gravado
    # uma vez) para o trace não perder essa informação.
    latency_ms = int((time.monotonic() - start) * 1000)
    update = {
        "grounded": grounded,
        "grounding_warnings": warnings,
        "response_final": draft,
        "verification_latency_ms": latency_ms,
    }
    if not state.get("initial_response_draft"):
        update["initial_response_draft"] = draft
        update["initial_grounded"] = grounded
        update["initial_grounding_warnings"] = warnings
    return update
