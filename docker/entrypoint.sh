#!/bin/bash
set -e

INDEX_FILE="data/processed/chroma_index/chroma.sqlite3"

if [ ! -f "$INDEX_FILE" ]; then
    echo "Índice ChromaDB não encontrado em $INDEX_FILE — construindo a partir do corpus..."
    python -m src.ingestion.build_index
fi

exec streamlit run interface/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501
