# Schema do Trace JSON

Cada execução do pipeline salva um arquivo JSON em `/traces/` com o seguinte schema.
Este documento é a referência canônica — implementação em `src/observability/tracer.py`.

## Campos raiz

| Campo | Tipo | Descrição |
|---|---|---|
| `trace_id` | string (UUID4) | Identificador único da execução |
| `query_original` | string | Pergunta exatamente como digitada pelo usuário |
| `timestamp` | string (ISO 8601) | Momento do início da execução (UTC) |
| `total_latency_ms` | int | Soma das latências de todos os agentes (ms) |
| `fallback_used` | bool | `true` se a busca web foi acionada (inicial ou corretiva) |
| `fallback_reason` | string \| null | Motivo do fallback (nulo se RAG foi suficiente) |
| `verifier_triggered_fallback` | bool | `true` se foi o *verifier* (não o `fallback_decision` inicial) quem acionou a busca web — caso "de fronteira": retrieval achou chunks acima do threshold, mas a resposta gerada a partir deles não era fundamentada. Ver `steps[].agent == "verifier".initial_verdict` para o veredito do primeiro passe (via RAG) nesse caso |
| `response_final` | string | Resposta entregue ao usuário |
| `grounded` | bool | Veredicto do agente verificador |
| `grounding_warnings` | list[string] | Afirmações sem suporte no contexto (vazio se grounded) |
| `sources` | list[string] | doc_ids ou URLs usados no contexto final |
| `steps` | list[StepObject] | Detalhamento por agente (ver abaixo) |

## StepObject — campos por agente

### `reformulator`
```json
{
  "agent": "reformulator",
  "input": "pergunta original",
  "output": ["reformulação 1", "reformulação 2"],
  "latency_ms": 420
}
```

### `retriever`
```json
{
  "agent": "retriever",
  "queries_used": ["original", "reform1", "reform2"],
  "chunks_retrieved": [
    {"chunk_id": "concepts/persistence_3", "similarity": 0.872, "title": "Persistence", "source_url": "https://..."}
  ],
  "fallback_triggered": false,
  "latency_ms": 210
}
```

### `fallback_decision`
```json
{
  "agent": "fallback_decision",
  "triggered": false,
  "reason": null,
  "latency_ms": 1
}
```

### `web_search` (apenas quando fallback_used=true)
```json
{
  "agent": "web_search",
  "corrective": false,
  "query": "pergunta original",
  "results": [
    {"title": "...", "url": "https://...", "score": 0.91}
  ],
  "latency_ms": 1200
}
```
`corrective: true` indica que este foi o fallback acionado pelo `verifier` (segundo passe), não pelo `fallback_decision` inicial.

### `generator`
```json
{
  "agent": "generator",
  "context_chunks_count": 3,
  "output_length_chars": 542,
  "latency_ms": 800
}
```
Se houve fallback corretivo, os campos refletem o **segundo** passe (via web) — o `context_used` final é sempre o que sustentou `response_final`.

### `verifier`
```json
{
  "agent": "verifier",
  "grounded": true,
  "warnings": [],
  "latency_ms": 350,
  "initial_verdict": null
}
```
`initial_verdict` só é preenchido quando o fallback corretivo rodou — contém `response_draft`/`grounded`/`warnings` do primeiro passe (via RAG), preservado porque `grounded`/`response_final` no nível raiz do trace são sobrescritos pelo segundo passe:
```json
{
  "agent": "verifier",
  "grounded": true,
  "warnings": [],
  "latency_ms": 340,
  "initial_verdict": {
    "response_draft": "Sim, o Command ganhou um parâmetro `graph` na 1.2...",
    "grounded": false,
    "warnings": ["- ... não fundamentado no contexto ..."]
  }
}
```
