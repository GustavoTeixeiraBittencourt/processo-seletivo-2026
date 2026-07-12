"""
Rodar: python -m src.retrieval._manual_check

Roda as perguntas de calibração contra o índice atual, imprime a distribuição
de similaridade top-1 por categoria e salva um relatório em
data/processed/similarity_calibration.json — evidência versionável para a
decisão de threshold em src/retrieval/retriever.py (MIN_SIMILARITY_DEFAULT),
em vez de só um comentário no código sem os números por trás.
"""
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval.retriever import MIN_SIMILARITY_DEFAULT, retrieve

REPORT_PATH = Path("data/processed/similarity_calibration.json")

PERGUNTAS_IN_CORPUS = [
    "Como funciona o checkpointing no LangGraph?",
    "O que é um subgraph e quando usar um?",
    "Como o LangGraph decide se um checkpoint deve ser persistido entre execuções?",
    "O que é o Pregel no contexto do LangGraph?",
    "Como funciona o Command no LangGraph 1.0?",
    "O que são checkpointers e para que servem?",
    "Como interromper a execução de um grafo com interrupt()?",
    "O que é a Graph API do LangGraph?",
]

PERGUNTAS_FALLBACK = [
    "O LangGraph 1.2 mudou a forma como o Command funciona comparado à 1.0?",
    "Quais são as novidades do LangGraph 1.3?",
    "Como o LangGraph se compara ao CrewAI em 2026?",
    "Qual é a capital da França?",
]

DECISAO = (
    "Threshold provisorio MIN_SIMILARITY_DEFAULT=0.78: separa bem perguntas "
    "in-corpus (top-1 tipicamente ~0.82-0.87) de perguntas fora de dominio "
    "(~0.70-0.71), mas ainda deixa passar fallback 'de fronteira' (mesmo "
    "dominio, fora do escopo temporal/versao do corpus, ~0.81-0.84) -- esses "
    "casos nao sao distinguiveis por similaridade de cosseno sozinha, porque "
    "o texto da pergunta e semanticamente proximo do conteudo indexado; a "
    "distincao exige raciocinio sobre o que o corpus cobre (fica a cargo do "
    "agente verificador, nao do retriever). Calibracao final fica pendente "
    "contra o split de validacao do benchmark (Dias 11-12), nao contra estas "
    "amostras informais."
)


def _collect(perguntas: list[str], top_k: int = 5) -> list[dict]:
    resultados = []
    for q in perguntas:
        hits = retrieve(q, top_k=top_k, min_similarity=0.0)
        resultados.append({
            "question": q,
            "hits": [
                {"similarity": round(h["similarity"], 4), "doc_id": h["metadata"]["doc_id"]}
                for h in hits
            ],
        })
    return resultados


def _summary(resultados: list[dict]) -> dict:
    top1 = [r["hits"][0]["similarity"] for r in resultados if r["hits"]]
    if not top1:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": round(min(top1), 4),
        "max": round(max(top1), 4),
        "mean": round(statistics.mean(top1), 4),
    }


def run_calibration() -> dict:
    in_corpus = _collect(PERGUNTAS_IN_CORPUS)
    fallback = _collect(PERGUNTAS_FALLBACK)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_similarity_threshold": MIN_SIMILARITY_DEFAULT,
        "in_corpus": {"summary_top1": _summary(in_corpus), "questions": in_corpus},
        "fallback": {"summary_top1": _summary(fallback), "questions": fallback},
        "decision": DECISAO,
    }


if __name__ == "__main__":
    report = run_calibration()
    for titulo, key in [
        ("IN-CORPUS (deveria retornar hits)", "in_corpus"),
        ("FALLBACK (deveria retornar poucos/nenhum hit)", "fallback"),
    ]:
        print(f"=== {titulo} ===")
        for r in report[key]["questions"]:
            print(r["question"])
            for h in r["hits"]:
                print(f"  {h['similarity']:.3f}  {h['doc_id']}")
        print(f"  resumo top-1: {report[key]['summary_top1']}")
        print()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Relatório salvo em {REPORT_PATH}")
