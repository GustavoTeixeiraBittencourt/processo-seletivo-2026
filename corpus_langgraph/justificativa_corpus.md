# Justificativas Corpus do projeto - LangGraph documentation

### 1- Por que esse corpus e o que ele representa como problema real

Pois é uma documentação de framework que, além de estarmos trabalhando em cima, vai nos ajudar no aprendizado continuo sobre Inteligencia Artificial. Outra questão é a validação da resposta com autoridade, visto que vamos arquitetar perguntas sobre o framework, facilitando a validação da resposta que foi gerada.

### 2- Como a combinação RAG e busca web resolve algo que nenhum dos dois sozinho resolve bem nesse domínio ?

O RAG irá cobrir a versão citada nos documentos com a citação e com precisão nos detalhes (1.0.0). já a busca na web cobre o que 
existe fora do tempo ou do escopo do projeto. Nenhum dos dois cobre as duas categorias da pergunta.

### 3.​ Onde o sistema vai falhar e por que isso é esperado dado o corpus escolhido

O sistema falha em duas categorias, por razões diferentes:

**(a) Cobertura de conteúdo** — perguntas sobre features/versões posteriores ao corte
da tag `1.0.0`, ou perguntas que exigem síntese entre múltiplas seções distantes do
corpus. Esperado: o corte de versão é a fronteira deliberada do projeto.

**(b) Qualidade da fonte de fallback** — mesmo quando o sistema classifica
corretamente que uma pergunta precisa de busca web (100% de acerto no benchmark,
ver `benchmark/results_20260714T161811.json`), a resposta final pode ficar
factualmente errada quando a busca web devolve fontes pouco relevantes (ex.
`fallback_04`: a Tavily retornou pacotes PyPI que não são o `langgraph` real) ou
quando o modelo de geração confunde detalhes entre entidades estruturalmente
parecidas no texto-fonte (ex `fallback_07`: atribuição cruzada de versão/descrição entre CVEs diferentes). Isso é esperado porque, ao contrário do corpus RAG — limpo, determinístico e sob controle do projeto — o conteúdo de fallback vem de uma fonte externa não curada. O agente verificador detecta os dois casos de forma reproduzível (`grounded: false` em duas rodadas de teste independentes), provando que a rede de segurança funciona; a resposta final permanece incorreta porque o loop de correção é limitado a uma tentativa por desenho (evitar latência/custo descontrolados), não por ausência de verificação.

### 4.​ Por que esse corpus permite exercitar todas as partes obrigatórias do desafio, incluindo o fallback ?
Pois o corte de data cria uma fronteira objetiva e verificável — não é "fallback fingido", é uma limitação real e documentável.
