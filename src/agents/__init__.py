"""Agentes do pipeline — cada módulo expõe uma função `run(state) -> dict`."""
from src.agents import (  # noqa: F401  (importados para que graph.py use `from src.agents import x`)
    fallback_decision,
    generator,
    reformulator,
    retriever,
    verifier,
    web_search,
)
