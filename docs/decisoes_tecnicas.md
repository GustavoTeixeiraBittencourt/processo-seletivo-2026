# Decisões técnicas

Registro das decisões de arquitetura que não são óbvias só lendo o código —
o quê foi decidido, por quê, e qual trade-off foi aceito. Alimenta a seção
de justificativas do README final.

## LLM de geração: Groq (`llama-3.1-8b-instant`), não Ollama local

**Decisão final:** o agente gerador, o reformulador e o verificador usam a
API da Groq na nuvem (`src/agents/_llm.py`), não um modelo rodando local via
Ollama.

**Contexto:** o hardware de desenvolvimento é um notebook com GPU MX450 (2GB
de VRAM). O plano original (documentado no guia de execução) considerava
rodar `qwen2.5:3b` localmente via Ollama, reservando a GPU pequena para esse
modelo e mantendo o modelo de embeddings (`intfloat/multilingual-e5-small`)
em CPU. Essa configuração chegou a ser implementada.

**Por que revertemos para Groq:**
- **Hardware:** cada execução do pipeline faz de 3 a 5 chamadas de LLM
  (reformulator, generator, verifier — até 2x cada se o loop de fallback
  corretivo disparar, ver `src/orchestration/graph.py`). Rodar isso local
  num modelo 3B numa GPU de 2GB (ou em CPU, se a GPU não comportar o
  contexto) compete com o resto do sistema e é sensivelmente mais lento —
  tanto para desenvolvimento iterativo (rodar o benchmark várias vezes
  durante calibração) quanto, potencialmente, para a demo ao vivo com o
  avaliador.
- **Tempo:** a Groq serve inferência com latência bem menor que hardware
  local limitado, o que importa porque o benchmark tem 20 perguntas e cada
  uma pode disparar múltiplas chamadas sequenciais.
- **Escopo do edital:** nada no `dados/desafio.pdf` exige que o LLM de
  geração rode localmente — a restrição de VRAM do guia era uma
  recomendação de projeto, não um requisito de avaliação.

**Trade-off aceito:** o pipeline passa a depender de rede e de uma
`GROQ_API_KEY` (free tier) para a etapa de geração. Isso é mitigado por
retry automático com backoff exponencial em rate limit (429) em
`call_llm()`. Essa dependência de API key **não afeta a embedding/indexação**
— `build_index.py` e `retriever.py` continuam rodando 100% local em CPU,
decisão independente e mantida sem alteração.

**Nota histórica:** o projeto trocou para Ollama por um período curto do
desenvolvimento, motivado pela leitura literal do guia de execução. Essa
troca foi revertida — a entrada aqui documenta a decisão final para não
haver ambiguidade em revisões futuras do código ou na entrevista.
