# Detector de Interacoes Medicamentosas com LLMs

**Disciplina:** Sistemas Cognitivos com Large Language Models
**Aluno:** Kevin Rodrigues
**Projeto:** Detector de Interacoes Medicamentosas em Bularios Farmaceuticos

---

## 1. Descricao do Problema

Aplicacao cognitiva que recebe consultas em linguagem natural sobre interacoes entre medicamentos e retorna classificacoes estruturadas (grave, leve/moderada, sem interacao) fundamentadas em bulas reais de medicamentos brasileiros.

**Problema real:** Profissionais de saude e pacientes precisam verificar interacoes medicamentosas rapidamente. Bularios sao extensos e buscar manualmente e ineficiente. O sistema extrai e classifica interacoes automaticamente.

---

## 2. Corpus ou Base de Conhecimento

| Fonte | Arquivos | Descricao |
|-------|----------|-----------|
| Fonte 1 (ANVISA) | 4.978 bulas | Bularios oficiais em formato `_paciente` / `_profissional`. Secao de interacoes extraida. |
| Fonte 2 (Consultaremedios) | 982 Q&A | Perguntas e respostas structuradas. Bloco `INTERACAO MEDICAMENTOSA?` e `COMPOSICAO?`. |
| **Total** | **5.960** | Processados e dobedores para `data/pruned/` |

Dados podados disponiveis em: `C:\workspace\python\python-processador-bulas\data\pruned\`

---

## 3. Arquitetura do Sistema

```
Consulta (linguagem natural)
  |
  v
[NER] pucpr/clinicalnerpt-chemical
  -> Extrai nomes de farmacos da consulta
  |
  v
[Embeddings] paraphrase-multilingual-MiniLM-L12-v2 (384d)
  -> Codifica trechos de bulas em vetores densos
  |
  v
[Indice FAISS] IndexFlatIP
  -> Recupera top-K trechos mais similares
  |
  v
[Classificador] GPT4All / Heuristica
  -> Few-shot classification (Classe 0, 1 ou 2)
  |
  v
[JSON estruturado]
  -> {"classe": N, "justificativa": "...", "evidencia": "..."}
```

---

## 4. Modelos Utilizados

| Modelo | Tarefa | Justificativa |
|--------|--------|---------------|
| `pucpr/clinicalnerpt-chemical` | NER | Treinado em bulas medicas PT-BR, vocabulario farmaceutico |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embeddings | Suporta PT-BR, 384 dimensoes, otimizado para similaridade |
| `Meta-Llama-3-8B-Instruct.Q4_0.gguf` (opcional) | Classificacao | GPT4All local, inferencia privada |
| Heuristica (fallback) | Classificacao | Palavras-chave (contraindicado, monitorar, etc.) |

---

## 5. Instalacao

### 5.1 Requisitos

- Python 3.11+
- CUDA (opcional, para GPU)

### 5.2 Instalacao de Dependencias

```bash
cd C:\workspace\python\projeto-2-modulo-1-pos
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 Requisitos (requirements.txt)

```
transformers<5
sentencepiece
sacrebleu
sentence-transformers
faiss-cpu
torch
gpt4all
openai
notebook
```

---

## 6. Executando os Notebooks

Cada notebook e independente e pode ser executado com "Run All":

```bash
# Selecionar kernel: venv com python-dotenv
jupyter notebook c01_modelos_llm.ipynb
jupyter notebook c02_engenharia_prompt.ipynb
jupyter notebook c03_embeddings_busca.ipynb
jupyter notebook c04_inferencia_local.ipynb
jupyter notebook c05_pipeline_rag.ipynb
```

---

## 7. Detalhamento dos Notebooks

### c01_modelos_llm.ipynb
- Pipeline NER com `pucpr/clinicalnerpt-chemical`
- Comparacao entre pipeline e inferencia manual
- Analise de saidas do modelo (B-I tagging, agregacao)

### c02_engenharia_prompt.ipynb
- **Tecnica 1:** Zero-shot prompting (classificacao direta)
- **Tecnica 2:** Few-shot prompting (exemplos no prompt)
- **Tecnica 3:** Chain-of-thought (raciocinio encadeado)
- Saida JSON validada com `json.loads()` + regex fallback

### c03_embeddings_busca.ipynb
- Geracao de embeddings com `sentence-transformers`
- Indice FAISS (IndexFlatIP - produto interno = cosseno normalizado)
- Metricas: recall@5, MRR@5
- Comparacao BM25 vs busca densa vs hibrida

### c04_inferencia_local.ipynb
- GPT4All direto (`gpt4all.GPT4All`)
- GPT4All via API local (`http://localhost:4891/v1`)
- Heuristica (fallback)
- Comparacao: privacidade, custo, latencia

### c05_pipeline_rag.ipynb
- Pipeline RAG completo integrado
- NER + recuperacao + classificacao em uma funcao
- Resposta com e sem contexto recuperado
- Analise de seguranca (injecao de prompt, vazamento)

---

## 8. Classes de Interacao

| Classe | Rotulo | Criterio |
|--------|--------|----------|
| 0 | `SEM_INTERACAO` | Nenhuma interacao relevante |
| 1 | `LEVE_MODERADA` | Monitorar, ajustar dose, cautela |
| 2 | `GRAVE_CONTRAINDICADA` | Contraindicado, risco de morte |

---

## 9. Estrategia de Execucao

**Privacidade:** 100% local. Dados medicos nunca saem da maquina.

| Estrategia | Custo | Latencia | Privacidade |
|-----------|-------|----------|------------|
| GPT4All direto | R$ 0 | Alta | Total |
| GPT4All API local | R$ 0 | Media | Alta |
| Heuristica | R$ 0 | Baixa | Total |
| OpenAI API | $ | Baixa | Baixa |

---

## 10. Pipeline RAG

### 10.1 Carregamento de Documentos
```python
# Fonte 1: bulas ANVISA (4.978 arquivos .txt)
# Fonte 2: Q&A Consultaremedios (982 arquivos .txt)
# Podados em: data/pruned/fonte1/ e data/pruned/fonte2/
```

### 10.2 Chunking
- Estrategia: sentencas curtas (max 250 palavras)
- Overlap: 1 sentenca entre chunks
- Garantia: tokenizador BERT nao corta informacao no meio

### 10.3 Indexacao
```python
indice_faiss = faiss.IndexFlatIP(384)
indice_faiss.add(matriz_embeddings.astype(np.float32))
```

### 10.4 Recuperacao
```python
# Busca por similaridade cosseno (IP com vetores normalizados)
distancias, indices = indice_faiss.search(embed_consulta, top_k=5)
```

### 10.5 Resposta com Contexto
- Prompt Few-shot com trechos recuperados
- Reduz alucinacao em comparacao com zero-shot puro

---

## 11. Seguranca

### 11.1 Riscos Identificados

| Risco | Descricao | Controle |
|-------|-----------|----------|
| Injecao de prompt | Consulta contem instrucoes para bypassar classificacao | Sanitizacao de entrada (regex "lembre_se", "ignore_instrucoes") |
| Vazamento de contexto | Tentativa de extrair system prompt | Log de auditoria; heuristica nao expõe prompt |
| Exposição de dados medicos | API externa recebe dados de pacientes | GPT4All local; dados nunca saem da maquina |

### 11.2 Controles Propostos
- Sanitizacao: regex remove blocos de injecao
- Heuristica fallback: funciona sem LLM
- Cache local: Redis TTL=7d para evitar reprocessamento

---

## 12. Limitações e Melhorias Futuras

### Limitacoes
- NER com vocabulario limitado (nomes comerciais fora do dominio)
- Heuristica com acuracia moderada (fallback only)
- Top-K pode perder contextos relevantes

### Melhorias
- Fine-tuning do classificador com as 5.960 bulas
- Reranking com segundo modelo
- Cache Redis para consultas frequentes
- Curadoria humana para casos ambíguos

---

## 13. Estrutura de Arquivos

```
projeto-2-modulo-1-pos/
  c01_modelos_llm.ipynb         # NER
  c02_engenharia_prompt.ipynb   # Prompting
  c03_embeddings_busca.ipynb    # Embeddings + FAISS
  c04_inferencia_local.ipynb    # GPT4All
  c05_pipeline_rag.ipynb        # Pipeline RAG completo
  requirements.txt
  README.md
  _gerar_nb0*.py               # Scripts geradores de notebook
  data/                        # Dados de bulas (opcional)
  logs/                        # Logs de execucao
  venv/                        # Ambiente virtual
```

---

## 14. Como Reproduzir

```bash
# 1. Clonar e configurar ambiente
cd C:\workspace\python\projeto-2-modulo-1-pos
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 2. Executar notebooks em ordem
jupyter notebook c01_modelos_llm.ipynb
# ... executar "Run All" em cada um

# 3. Testar pipeline RAG
# Abrir c05_pipeline_rag.ipynb e executar "Run All"
# GPT4All (opcional): baixar Meta-Llama-3-8B-Instruct.Q4_0.gguf
# Heuristica funciona sem modelo GGUF
```

---

## 15. Rubricas Atendidas

| Rubrica | Itens Demonstrados |
|---------|---------------------|
| 1. NLP com LLMs | NER (clinicalnerpt), pipeline transformers, comparacao de arquiteturas |
| 2. Prompt Engineering | Zero-shot, few-shot, chain-of-thought, saida JSON validada |
| 3. Embeddings e Busca | sentence-transformers, FAISS, busca hibrida BM25+densa |
| 4. Inferencia Local | GPT4All direto, API local, heuristica; comparacao de privacidade/custo |
| 5. Pipeline RAG | Carregamento, chunking, embeddings, recuperacao, geracao, seguranca |
