"""Fábrica de LLM — ponto único de configuração do modelo de linguagem.

Provider: Groq (nuvem), servindo llama-3.1-8b-instant. Decisão revertida de
volta de Ollama local (qwen2.5:3b) — ver justificativa completa em
docs/decisoes_tecnicas.md e no README. Resumo: mesmo com a GPU do notebook
tendo só 2GB de VRAM (o que motivou considerar um modelo local pequeno), a
Groq oferece inferência mais rápida que o hardware local conseguiria e
libera a GPU inteira para outras tarefas — o custo é depender de rede e de
uma API key, aceitável para o escopo deste desafio.

Isola a escolha de provider do restante do código: trocar de Groq para
outro provider requer mudar apenas este arquivo e as variáveis de ambiente.

Retry automático em rate limit (HTTP 429) via call_llm() — todos os agentes
devem usar call_llm(messages) em vez de get_llm().invoke(messages) para
aproveitar o backoff sem duplicar lógica por agente.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.1-8b-instant"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # segundos; espera será 2^(tentativa+1): 2s, 4s, 8s


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    # `or` (não getenv(key, default)) porque .env.example deixa GROQ_MODEL em
    # branco de propósito — "" é falsy e cai no default, mas
    # getenv(key, default) só aplicaria o default se a chave estivesse
    # ausente, não vazia
    model = os.getenv("GROQ_MODEL") or _DEFAULT_MODEL
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY não encontrada. "
            "Crie o arquivo .env com base em .env.example."
        )
    return ChatGroq(model=model, api_key=api_key, temperature=0)


def call_llm(messages: list[BaseMessage]) -> BaseMessage:
    """Invoca o LLM com retry automático em caso de rate limit (HTTP 429).

    Todos os agentes usam esta função — a lógica de retry fica centralizada
    aqui e não se repete em cada arquivo de agente.
    """
    llm = get_llm()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            exc_text = str(exc).lower()
            is_rate_limit = (
                "429" in str(exc)
                or "rate_limit" in exc_text
                or "rate limit" in exc_text
                or "too many requests" in exc_text
            )
            if not is_rate_limit:
                raise  # erro não é rate limit — propaga imediatamente

            last_exc = exc
            wait = float(_BACKOFF_BASE ** (attempt + 1))  # 2, 4, 8 s

            # Groq retorna retry-after em e.response.headers quando disponível
            try:
                retry_after = exc.response.headers.get("retry-after")
                if retry_after:
                    wait = max(wait, float(retry_after))
            except AttributeError:
                pass

            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "Rate limit Groq (429) — tentativa %d/%d — aguardando %.0fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)

    raise last_exc  # type: ignore[misc]
