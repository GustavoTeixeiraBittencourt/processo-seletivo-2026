j# Manifesto do Corpus — LangGraph Docs

## Proveniência (necessário para reprodutibilidade do pipeline de ingestão)

- Repositório: https://github.com/langchain-ai/langgraph
- Tag/versão fixada: `1.0.0`
- Commit hash exato: `c4144bb48f97e6afeaa002de318580cc41c19893`
- Data de download: 06/07/2026
- Método de download via: `codeload.github.com/langchain-ai/langgraph/tar.gz/refs/tags/1.0.0`

## Por que a versão 1.0.0 (e não a doc atual do site)

A tag `1.0.0` é a última fonte estável, versionada e imutável de conteúdo real:
o mesmo commit sempre vai gerar o mesmo conteúdo, o que é exatamente o
requisito "mesmo corpus → mesmo índice" do edital.

Isso também **fortalece o argumento de fallback**: qualquer pergunta sobre
recursos adicionados depois da 1.0.0 (a LangGraph já está na 1.2.x) ou sobre
qualquer conteúdo hospedado em docs.langchain.com não tem resposta no corpus
por construção — não é um fallback forçado, é uma fronteira de versão real e
verificável.

## Estatísticas do corpus

- 20 arquivos Markdown
- Pós-limpeza: **63.954 tokens** (`tiktoken` `cl100k_base` como proxy), bem acima
  do mínimo do edital (20 documentos OU 50.000 tokens) — ver "Estado pós-limpeza"
  abaixo para o detalhamento de como o gap inicial foi fechado.

## Estado pós-limpeza (`src/ingestion/cleaning.py`)

A doc do LangGraph pareia todo exemplo em blocos `:::python` / `:::js`
(conteúdo quase idêntico em Python e em JS/TS). Como este projeto é 100%
Python, a limpeza descarta por completo o conteúdo `:::js`/`:::ts` em vez de
manter as duas variantes sem rótulo (o que geraria duplicação e ruído
semântico nos embeddings). Isso é a decisão correta de qualidade, mas reduziu
bastante o volume inicial: **20 documentos, 63.954 tokens** (`cl100k_base`), acima do mínimo nos dois
critérios com folga.

## Lista de arquivos e URLs de origem (raw, fixados no commit)

- `docs/concepts/why-langgraph.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/why-langgraph.md
- `docs/concepts/low_level.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/low_level.md
- `docs/concepts/agentic_concepts.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/agentic_concepts.md
- `docs/concepts/multi_agent.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/multi_agent.md
- `docs/concepts/persistence.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/persistence.md
- `docs/concepts/memory.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/memory.md
- `docs/concepts/human_in_the_loop.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/human_in_the_loop.md
- `docs/concepts/streaming.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/streaming.md
- `docs/concepts/subgraphs.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/subgraphs.md
- `docs/concepts/functional_api.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/functional_api.md
- `docs/concepts/durable_execution.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/durable_execution.md
- `docs/concepts/time-travel.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/time-travel.md
- `docs/concepts/tools.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/tools.md
- `docs/concepts/pregel.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/pregel.md
- `docs/agents/overview.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/agents/overview.md
- `docs/agents/multi-agent.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/agents/multi-agent.md
- `docs/agents/context.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/agents/context.md
- `docs/tutorials/rag/langgraph_agentic_rag.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/tutorials/rag/langgraph_agentic_rag.md
- `docs/tutorials/get-started/1-build-basic-chatbot.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/tutorials/get-started/1-build-basic-chatbot.md
- `docs/how-tos/graph-api.md` — https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/how-tos/graph-api.md (adicionado em 2026-07-07 para fechar o gap de tamanho mínimo do corpus)
