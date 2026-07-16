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

## Recuperação híbrida (BM25 + RRF): threshold denso continua sendo o único gate

**Decisão final:** `src/agents/retriever.py` combina busca densa (E5) com
BM25 via Reciprocal Rank Fusion (`src/retrieval/hybrid.py`), mas o BM25
**nunca admite um chunk que não tenha passado no threshold de similaridade
denso** (`MIN_SIMILARITY_DEFAULT = 0.78`, `src/retrieval/retriever.py`). O
BM25 só reordena/reforça os candidatos que a busca densa já qualificou para
cada query (original + reformulações).

**Contexto:** `src/retrieval/hybrid.py` (`build_bm25`,
`reciprocal_rank_fusion`) já existia desde a fase de calibração da busca
densa, mas ficou deliberadamente não integrado até essa calibração estar
concluída — ver a entrada de threshold abaixo e `README.md`. Com o threshold
0.78 já validado contra o benchmark (100% de acurácia na decisão de
fallback), era hora de integrar.

**Por que o threshold denso continua sendo o gate, em vez de deixar o BM25
também admitir candidatos:**
- O threshold 0.78 foi calibrado especificamente para separar perguntas
  fora do domínio (`~0.70`) de perguntas do corpus (`~0.82-0.87`) — é o que
  sustenta os 100% de acurácia de fallback no benchmark. Deixar o BM25
  admitir chunks que a busca densa rejeitou reabriria essa calibração sem
  nova evidência: uma pergunta fora do domínio pode ter sobreposição lexical
  incidental com o corpus (uma palavra comum) sem ter, de fato, resposta
  nele — isso quebraria a garantia de fallback, não a melhoraria.
- O ganho esperado do BM25 aqui é outro: perguntas com termos exatos (nomes
  de função/classe, ex. `add_conditional_edges`, `interrupt()`) que o
  embedding denso às vezes rankeia mais abaixo do que deveria, mas que ainda
  assim aparecem no top-`_DENSE_CANDIDATES_PER_QUERY` (10) da busca densa —
  o BM25 empurra esses chunks para cima na fusão final. Isso é reordenação
  de precisão dentro do conjunto já aprovado, não expansão de recall além
  do threshold.

**Design da fusão:** para cada query (original + reformulações), busca-se
até 10 candidatos densos (acima do threshold) e até 10 candidatos BM25;
os dois rankings são fundidos por RRF (`reciprocal_rank_fusion`). Os
rankings fundidos de cada query são então combinados entre si por uma
segunda rodada de RRF (mesma fórmula, agora entre queries). Um filtro final
garante que só chunk_ids presentes no conjunto denso-qualificado (de
qualquer uma das queries) sobrevivem — o BM25 nunca introduz um chunk novo
sozinho. Cada chunk retornado carrega `matched_by: "hybrid"` (também achado
pelo BM25) ou `"dense"` (só a busca densa), visível no trace
(`docs/trace_schema.md`) para auditoria.

**Trade-off aceito:** o índice BM25 é reconstruído em memória por processo
(memoizado com `lru_cache`, mesmo padrão do modelo de embeddings) a partir
de todos os chunks do Chroma — custo aceitável no tamanho atual do corpus
(183 chunks), mas não escalaria sem paginação para um corpus ordens de
grandeza maior.
