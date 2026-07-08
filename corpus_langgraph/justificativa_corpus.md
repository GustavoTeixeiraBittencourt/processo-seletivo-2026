# Justificativas Corpus do projeto - LangGraph documentation

### 1- Por que esse corpus e o que ele representa como problema real

Pois é uma documentação de framework que, além de estarmos trabalhando em cima, vai nos ajudar no aprendizado continuo sobre Inteligencia Artificial. Outra questão é a validação da resposta com autoridade, visto que vamos arquitetar perguntas sobre o framework, facilitando a validação da resposta que foi gerada.

### 2- Como a combinação RAG e busca web resolve algo que nenhum dos dois sozinho resolve bem nesse domínio ?

O RAG irá cobrir a versão citada nos documentos com a citação e com precisão nos detalhes (1.0.0). já a busca na web cobre o que 
existe fora do tempo ou do escopo do projeto. Nenhum dos dois cobre as duas categorias da pergunta.
### 3.​ Onde o sistema vai falhar e por que isso é esperado dado o corpus escolhido

Irá falhar em perguntas sobre feature/versões que não estão no documentos pois excede a data limite do documentação ou perguntas distantes da documentação que exijam síntese entre multíplas seções diferentes.

### 4.​ Por que esse corpus permite exercitar todas as partes obrigatórias do desafio, incluindo o fallback ?
Pois o corte de data cria uma fronteira objetiva e verificável — não é "fallback fingido", é uma limitação real e documentável.