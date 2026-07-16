FROM python:3.12-slim

WORKDIR /app

# Dependências de compilação necessárias por sentence-transformers e chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependências Python ────────────────────────────────────────────────────
# 1. PyTorch CPU-only primeiro — requirements.txt contém a versão GPU (torch==2.13.0
#    compilada com CUDA) que não instala em containers sem driver NVIDIA.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Demais pacotes: exclui CUDA/NVIDIA/Triton (GPU-only) e torch (já instalado).
COPY requirements.txt .
RUN grep -vE '^(cuda-|nvidia-|triton|torch)' requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# ── Código da aplicação ────────────────────────────────────────────────────
COPY src/           ./src/
COPY interface/     ./interface/
COPY docs/          ./docs/
COPY .streamlit/    ./.streamlit/
COPY corpus_langgraph/ ./corpus_langgraph/

# Apenas o arquivo estático de pares QA — os resultados de benchmark são
# gerados em runtime e ficam no volume montado.
COPY benchmark/qa_pairs.json ./benchmark/qa_pairs.json

# Diretórios de saída — substituídos pelos volumes do compose em produção.
RUN mkdir -p data/processed/chroma_index traces

# ── Modelo de embeddings ───────────────────────────────────────────────────
# Pré-download em tempo de build: elimina o cold start de ~30 s na primeira
# query, que envolve baixar e carregar intfloat/multilingual-e5-small.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small', device='cpu'); \
print('Modelo de embeddings pronto.')"

# ── Entrypoint ─────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

# Streamlit expõe /_stcore/health após inicializar — start-period alto porque
# o LangGraph e o ChromaDB levam tempo para carregar na primeira requisição.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
