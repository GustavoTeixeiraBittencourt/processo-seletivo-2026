"""Fábrica de LLM — ponto único de configuração do modelo de linguagem.

Isola a escolha de provider do restante do código: trocar de Claude para
GPT requer mudar apenas este arquivo e a variável de ambiente LLM_MODEL.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY não encontrada. "
            "Crie o arquivo .env com base em .env.example."
        )
    return ChatAnthropic(model=model, api_key=api_key, temperature=0)
