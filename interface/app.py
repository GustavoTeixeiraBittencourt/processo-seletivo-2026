"""Interface Streamlit do sistema agêntico RAG (LangGraph docs + fallback web).

Único ponto de integração com o backend: `run_pipeline(question)`, que já
cuida de reformulação, recuperação, decisão de fallback, busca web,
geração e verificação, retornando a resposta final junto com o trace
completo (ver docs/trace_schema.md). Esta interface não implementa nenhuma
lógica de RAG/agentes — só consome e exibe o que o pipeline já produz.

Rodar com: streamlit run interface/app.py (a partir da raiz do repo).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Streamlit roda o script tendo o diretório do próprio arquivo como base,
# não a raiz do repo — sem isso `from src...` quebra.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv()

from src.orchestration.graph import run_pipeline  # noqa: E402

BENCHMARK_DIR = ROOT / "benchmark"
QA_PATH = BENCHMARK_DIR / "qa_pairs.json"
CORPUS_JUSTIFICATION_PATH = ROOT / "corpus_langgraph" / "justificativa_corpus.md"

st.set_page_config(page_title="LangGraph RAG Assistant", layout="wide")

st.markdown(
    """
    <style>
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.35rem;
    }
    .badge-neutral { background-color: #1A1A1A; color: #F7F1E3; }
    .badge-accent { background-color: #8A5A38; color: #F7F1E3; }
    .answer-card {
        background-color: #EDE3CE;
        border-left: 4px solid #8A5A38;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _badge_html(text: str, kind: str) -> str:
    css_class = "badge-accent" if kind == "accent" else "badge-neutral"
    return f'<span class="badge {css_class}">{text}</span>'


def _run_query(question: str):
    try:
        response_final, trace, trace_path = run_pipeline(question)
        return True, (response_final, trace, trace_path)
    except Exception as exc:  # noqa: BLE001 — nunca deixar a UI quebrar ao vivo
        return False, str(exc)


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _load_qa_pairs() -> dict:
    pairs = _load_json(str(QA_PATH))
    return {p["id"]: p for p in pairs}


def _latest_benchmark_file() -> Path | None:
    files = sorted(BENCHMARK_DIR.glob("results_*.json"))
    return files[-1] if files else None


def _render_steps(steps: list[dict]) -> None:
    for step in steps:
        agent = step.get("agent", "")
        st.markdown(f"**{agent}** · {step.get('latency_ms', 0)} ms")

        if agent == "reformulator":
            reformulations = step.get("output", [])
            if reformulations:
                for r in reformulations:
                    st.markdown(f"- {r}")
            else:
                st.caption("Nenhuma reformulação gerada.")

        elif agent == "retriever":
            chunks = step.get("chunks_retrieved", [])
            if chunks:
                st.dataframe(pd.DataFrame(chunks), width="stretch", hide_index=True)
            else:
                st.caption("Nenhum chunk acima do threshold de similaridade.")

        elif agent == "fallback_decision":
            st.write("Acionado:", "Sim" if step.get("triggered") else "Não")
            if step.get("reason"):
                st.caption(step["reason"])

        elif agent == "web_search":
            label = "Busca corretiva (pós-verificação)" if step.get("corrective") else "Busca web (fallback inicial)"
            st.write(label)
            results = step.get("results", [])
            if results:
                st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)

        elif agent == "generator":
            st.write(
                f"Chunks de contexto usados: {step.get('context_chunks_count', 0)}"
                f" · tamanho da resposta: {step.get('output_length_chars', 0)} caracteres"
            )

        elif agent == "verifier":
            st.write("Fundamentado:", "Sim" if step.get("grounded") else "Não")
            for w in step.get("warnings", []):
                st.caption(f"Aviso: {w}")
            initial_verdict = step.get("initial_verdict")
            if initial_verdict:
                st.markdown("---")
                st.caption("Veredito da 1ª tentativa (via RAG), antes do fallback corretivo:")
                st.write(initial_verdict.get("response_draft", ""))
                st.write(
                    "Fundamentado (1ª tentativa):",
                    "Sim" if initial_verdict.get("grounded") else "Não",
                )

        st.divider()


def _render_sources(sources: list[str]) -> None:
    if not sources:
        st.caption("Nenhuma fonte registrada.")
        return
    for s in sources:
        if s.startswith("http"):
            st.markdown(f"- [{s}]({s})")
        elif s:
            st.markdown(f"- `{s}`")


def _render_result(response_final: str, trace: dict, show_details: bool = True) -> None:
    st.markdown(f'<div class="answer-card">{response_final}</div>', unsafe_allow_html=True)

    badges = []
    if trace.get("fallback_used"):
        badges.append(_badge_html("Fonte: Busca Web (fallback)", "accent"))
    else:
        badges.append(_badge_html("Fonte: RAG (corpus)", "neutral"))
    if trace.get("grounded"):
        badges.append(_badge_html("Fundamentado: Sim", "neutral"))
    else:
        badges.append(_badge_html("Fundamentado: Não", "accent"))
    badges.append(_badge_html(f"Latência total: {trace.get('total_latency_ms', 0)} ms", "neutral"))
    st.markdown(" ".join(badges), unsafe_allow_html=True)

    if trace.get("fallback_used") and trace.get("fallback_reason"):
        st.caption(f"Motivo do fallback: {trace['fallback_reason']}")

    if not show_details:
        return

    with st.expander("Etapas dos agentes", expanded=False):
        _render_steps(trace.get("steps", []))

    with st.expander("Fontes"):
        _render_sources(trace.get("sources", []))

    with st.expander("Trace JSON completo"):
        st.json(trace)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### LangGraph RAG Assistant")
    st.caption("Corpus: documentação do LangGraph, tag `1.0.0`")
    st.divider()
    st.markdown("**Status das chaves de API**")

    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    st.markdown(f"- GROQ_API_KEY: {'configurada' if groq_ok else 'ausente'}")
    st.markdown(f"- TAVILY_API_KEY: {'configurada' if tavily_ok else 'ausente'}")

    if not groq_ok:
        st.error("GROQ_API_KEY ausente — o pipeline não vai funcionar. Configure o .env.")
    elif not tavily_ok:
        st.warning("TAVILY_API_KEY ausente — perguntas que exigem fallback web vão falhar.")


# --------------------------------------------------------------------------
# Abas
# --------------------------------------------------------------------------

tab_consulta, tab_benchmark, tab_sobre = st.tabs(["Consulta", "Benchmark", "Sobre o Sistema"])

with tab_consulta:
    st.subheader("Pergunte sobre a documentação do LangGraph (tag 1.0.0)")

    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Pergunta",
            placeholder="Ex.: Como funciona checkpointing em um grafo compilado?",
            height=100,
        )
        submitted = st.form_submit_button("Perguntar")

    if submitted and question.strip():
        with st.spinner("Executando pipeline multiagente (reformulação → recuperação → geração → verificação)..."):
            ok, result = _run_query(question.strip())

        if not ok:
            st.error(f"Falha ao executar o pipeline: {result}")
        else:
            response_final, trace, _trace_path = result
            st.session_state.setdefault("history", [])
            st.session_state["history"].insert(
                0, {"question": question.strip(), "response": response_final, "trace": trace}
            )

    history = st.session_state.get("history", [])
    if history:
        latest = history[0]
        st.markdown(f"**Pergunta:** {latest['question']}")
        _render_result(latest["response"], latest["trace"], show_details=True)

        if len(history) > 1:
            st.markdown("### Histórico da sessão")
            for item in history[1:]:
                with st.expander(item["question"]):
                    _render_result(item["response"], item["trace"], show_details=False)
    else:
        st.info("Faça uma pergunta acima para ver a resposta e o trace completo do pipeline.")


with tab_benchmark:
    st.subheader("Resultados do benchmark (conjunto de teste)")

    latest_path = _latest_benchmark_file()
    if latest_path is None:
        st.warning("Nenhum resultado de benchmark encontrado em benchmark/.")
    else:
        data = _load_json(str(latest_path))
        summary = data["summary"]
        st.caption(f"Execução: {data['run_at']} · split: {data['split']} · arquivo: {latest_path.name}")

        def _pct(v):
            return f"{v:.0%}" if v is not None else "N/A"

        cols = st.columns(5)
        cols[0].metric("Similaridade semântica média", _pct(summary.get("avg_semantic_similarity")))
        cols[1].metric("Acerto classificação fallback", _pct(summary.get("fallback_classification_accuracy")))
        cols[2].metric("Doc recall", _pct(summary.get("doc_recall_rate")))
        cols[3].metric("Taxa de fundamentação", _pct(summary.get("grounded_rate")))
        avg_latency = summary.get("avg_latency_ms")
        cols[4].metric("Latência média", f"{avg_latency} ms" if avg_latency is not None else "N/A")

        qa_pairs = _load_qa_pairs()
        rows = []
        for r in data["results"]:
            pair = qa_pairs.get(r["id"], {})
            trace = None
            if r.get("trace_path"):
                trace_full_path = ROOT / r["trace_path"]
                if trace_full_path.exists():
                    trace = _load_json(str(trace_full_path))

            rows.append(
                {
                    "id": r["id"],
                    "pergunta": pair.get("question", ""),
                    "RAG ou fallback": "Fallback (Web)" if r.get("fallback_triggered") else "RAG",
                    "fonte recuperada": ", ".join(trace.get("sources", [])) if trace else "",
                    "resposta gerada": trace.get("response_final", "") if trace else "",
                    "pontuação (similaridade)": r.get("semantic_similarity"),
                    "fallback esperado": r.get("is_fallback_expected"),
                    "fallback correto": r.get("fallback_correct"),
                    "fundamentado": r.get("grounded"),
                    "latência (ms)": r.get("total_latency_ms"),
                }
            )

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


with tab_sobre:
    st.subheader("Corpus")
    if CORPUS_JUSTIFICATION_PATH.exists():
        st.markdown(_load_text(str(CORPUS_JUSTIFICATION_PATH)))
    else:
        st.caption("Justificativa do corpus não encontrada.")

    st.subheader("Arquitetura multiagente")
    agents_info = [
        ("reformulator", "Reformula a pergunta original em variações para melhorar a recuperação (multi-query)."),
        (
            "retriever",
            "Busca no vector store (Chroma) para a pergunta original e cada reformulação, "
            "mesclando os resultados pela maior similaridade.",
        ),
        (
            "fallback_decision",
            "Decide, de forma determinística, se aciona a busca web — dispara quando a "
            "recuperação não retorna nenhum chunk acima do threshold de similaridade.",
        ),
        (
            "web_search",
            "Busca na web via Tavily quando o corpus não tem a resposta (fallback inicial) "
            "ou quando o verificador rejeita a resposta do RAG (fallback corretivo).",
        ),
        ("generator", "Monta a resposta final a partir do contexto recuperado (RAG ou web), citando as fontes usadas."),
        (
            "verifier",
            "Verifica se a resposta está fundamentada no contexto recuperado; se não estiver e a "
            "busca web ainda não tiver sido tentada, aciona uma segunda passada via web_search.",
        ),
    ]
    for name, desc in agents_info:
        st.markdown(f"**{name}** — {desc}")

    st.subheader("Stack técnica")
    st.markdown(
        "- **Vector store:** ChromaDB (`PersistentClient`, similaridade cosseno)\n"
        "- **Embeddings:** `intfloat/multilingual-e5-small`, rodando em CPU\n"
        "- **Geração / reformulação / verificação:** Groq `llama-3.1-8b-instant` — decisão "
        "documentada em `docs/decisoes_tecnicas.md` (substituiu um plano inicial com Ollama "
        "local, revertido por limitação de hardware)\n"
        "- **Busca web (fallback):** Tavily"
    )
