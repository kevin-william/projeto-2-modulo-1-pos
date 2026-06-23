# Plano de Reboot — Detector de Interações Medicamentosas

**Versão:** 3.0 (reboot — arquitetura simplificada)  
**Projeto:** Módulo 1 — Sistemas Cognitivos com Large Language Models  
**Aluno:** Kevin Rodrigues  

---

## Sumário

1. [Diagnóstico do Projeto Atual](#1-diagnóstico-do-projeto-atual)
2. [Princípios do Reboot](#2-princípios-do-reboot)
3. [Nova Arquitetura](#3-nova-arquitetura)
4. [Estrutura de Arquivos](#4-estrutura-de-arquivos)
5. [Fluxo Completo do Projeto](#5-fluxo-completo-do-projeto)
6. [Notebook 01 — Modelos LLM e NLP](#6-notebook-01--modelos-llm-e-nlp)
7. [Notebook 02 — Prompt Engineering](#7-notebook-02--prompt-engineering)
8. [Notebook 03 — Embeddings e Busca Vetorial](#8-notebook-03--embeddings-e-busca-vetorial)
9. [Notebook 04 — Inferência Local vs Remota](#9-notebook-04--inferência-local-vs-remota)
10. [Notebook 05 — Pipeline RAG](#10-notebook-05--pipeline-rag)
11. [README + Relatório PDF](#11-readme--relatório-pdf)
12. [Mapeamento Rubricas → Células](#12-mapeamento-rubricas--células)
13. [Plano de Execução](#13-plano-de-execução)

---

## 1. Diagnóstico do Projeto Atual

### O que foi construído (v2.0)

| Camada | Arquivos | Linhas |
|---|---|---|
| `scripts/` | 5 arquivos (config, preprocess, annotate, validate, build) | ~1.500 |
| `docs/` | 13 arquivos (plano, 10 fases, 3 guias) | ~80 KB |
| `tests/` | 1 arquivo (17 testes) | ~250 |
| Notebooks | c01 (26 células, construído programaticamente) | 233 KB |
| `data/` | chunks JSONL (270k linhas), CSVs de anotação | ~50 MB |

### Problemas identificados

1.  **Over-engineering.** Construímos um pipeline de anotação com weak supervision,
    validação automática e exportação balanceada. O professor quer ver **compreensão
    de conceitos**, não um sistema de produção.

2.  **Fragmentação excessiva.** `config.py` (150 linhas), `preprocess.py` (330 linhas),
    `annotate.py` (400 linhas) — todos poderiam ser células de notebook. O aluno
    passa mais tempo navegando entre arquivos do que demonstrando conhecimento.

3.  **Construção programática de notebooks.** `_build_nb.py` gera `.ipynb` via
    `nbformat`. Isso é frágil (quebrou com transformers 5.x), difícil de manter,
    e completamente desnecessário — notebooks são feitos para serem escritos
    diretamente no Jupyter.

4.  **Dependência de dados externos.** O pré-processamento lê de
    `python-processador-bulas/data/pruned/`. Se esse caminho mudar, tudo quebra.

5.  **Complexidade desnecessária no armazenamento.** ChromaDB, JSONL com 270k
    linhas, CSVs de anotação — para um protótipo que só precisa demonstrar o
    conceito, FAISS em memória ou `sentence-transformers` + `cosine_similarity`
    são suficientes.

### O que FUNCIONA e deve ser mantido

-   **Conhecimento do dataset.** Sabemos exatamente como as bulas são estruturadas
    (Fonte 1: seções `##`, Fonte 2: blocos `[P: ...] R: ...`).
-   **Dados pré-processados.** O `python-processador-bulas` já podou as bulas —
    podemos usar esses arquivos diretamente.
-   **Modelos certos.** `clinicalnerpt-chemical` para NER, `bert-base-portuguese-cased`
    para embeddings, `biobertpt-all` como referência.
-   **Notebook 01 já executando.** 26 células, 6 pipelines, executou sem erro.

---

## 2. Princípios do Reboot

1.  **Cada notebook é autossuficiente.** Nada de `from scripts.config import ...`.
    Cada notebook importa o que precisa nas primeiras células. Se uma constante é
    usada em 2 notebooks, ela é definida em ambos. Duplicação consciente > acoplamento.

2.  **Dados embutidos ou carregados diretamente.** Nada de pipeline de pré-processamento
    separado. O notebook que precisa de chunks lê os arquivos `.txt` podados
    diretamente do disco.

3.  **Menos arquivos, mais células.** O que era um script de 400 linhas vira 3-4
    células de notebook com output visível.

4.  **Sem build scripts.** Notebooks são escritos e executados no Jupyter. Fim.

5.  **Simplicidade nas dependências.** Sem ChromaDB, sem `rank-bm25`, sem `gpt4all`
    (a menos que estritamente necessário). FAISS para busca vetorial. OpenAI API
    para LLM remoto.

---

## 3. Nova Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                   USUÁRIO (consulta NL)                         │
│          "Posso tomar Amoxicilina com Ibuprofeno?"             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              c05_rag_pipeline.ipynb (orquestrador)              │
│                                                                  │
│  ETAPA 1: NER (clinicalnerpt-chemical)                          │
│    "Amoxicilina", "Ibuprofeno"                                  │
│                                                                  │
│  ETAPA 2: Busca vetorial (FAISS + embeddings BERT pt)           │
│    Top-5 chunks com ambas as entidades                          │
│                                                                  │
│  ETAPA 3: Classificação (zero-shot via LLM com prompt)          │
│    Classe + confiança + evidência                               │
│                                                                  │
│  ETAPA 4: Resposta fundamentada (LLM + chunks)                  │
│    "A bula da Amoxicilina menciona que com Ibuprofeno..."       │
│                                                                  │
│  SAÍDA: JSON estruturado                                        │
└─────────────────────────────────────────────────────────────────┘

DADOS (lidos sob demanda, sem pré-processamento separado):
  data/bulas/fonte1/*.txt  (4.978 bulas ANVISA podadas)
  data/bulas/fonte2/*.txt  (982 bulas Consultaremedios podadas)
```

### Stack (drasticamente reduzida)

| Antes (v2.0) | Depois (v3.0) |
|---|---|
| `scripts/config.py` (150 linhas) | Constantes inline no notebook |
| `scripts/preprocess.py` (330 linhas) | Célula no notebook 03 e 05 |
| `scripts/annotate.py` (400 linhas) | **Removido** (sem anotação) |
| `scripts/validate_annotations.py` | **Removido** |
| `scripts/ner.py`, `classifier.py`, `rag.py` | **Removidos** (tudo inline) |
| ChromaDB persistente | FAISS em memória |
| `data/chunks_bulas.jsonl` (270k linhas) | Leitura direta dos `.txt` |
| `data/anotacoes/*.csv` (4 arquivos) | **Removido** |
| `_build_nb.py` | **Removido** |
| `docs/fases/FASE_0.md ... FASE_9.md` | **Removidos** |
| `docs/GUIA_ANOTACAO.md` | **Removido** |
| `docs/VALIDACAO_ANOTACOES.md` | **Removido** |
| `docs/PASSO_A_PASSO_ANOTACAO.md` | **Removido** |

**Total:** ~20 arquivos e ~1.500 linhas de script removidos.

---

## 4. Estrutura de Arquivos

```
C:\workspace\python\projeto-2-modulo-1-pos\   ← NOVO diretório limpo
├── c01_modelos_llm.ipynb
├── c02_prompting.ipynb
├── c03_embeddings_busca.ipynb
├── c04_inferencia_local_ou_remota.ipynb
├── c05_rag_pipeline.ipynb
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── bulas/                    ← Copiar de python-processador-bulas/data/pruned/
│       ├── fonte1/               ← 4.978 .txt
│       └── fonte2/               ← 982 .txt
└── docs/
    └── PLANO_REBOOT.md           ← Este arquivo
```

**Apenas 12 arquivos para entregar** (5 notebooks + 5 auxiliares + 1 doc + 1 dir de dados).

---

## 5. Fluxo Completo do Projeto

### Como o professor vai avaliar

1.  Abre `README.md` → instala dependências → copia dados
2.  Abre `c01` → vê 6 pipelines HuggingFace com células Markdown explicativas → **Rubrica 1 ✅**
3.  Abre `c02` → vê 3 técnicas de prompting + parsing JSON + iteração → **Rubrica 2 ✅**
4.  Abre `c03` → vê embeddings gerados + FAISS + busca híbrida + análise de falhas → **Rubrica 3 ✅**
5.  Abre `c04` → vê comparação OpenAI vs GPT4All em 5 dimensões → **Rubrica 4 ✅**
6.  Abre `c05` → vê pipeline RAG completo: NER → busca → LLM → JSON → **Rubrica 5 ✅**
7.  Lê o relatório PDF → vê todas as 26 seções obrigatórias preenchidas

### Dependência entre notebooks

```
c01 (HF + NLP)        independente
c02 (Prompting)        independente (usa API, não dados)
c03 (Embeddings)       independente (usa bulas .txt)
c04 (Inferência)       independente (usa API + GPT4All)
c05 (RAG Pipeline)     ── usa conhecimento dos 4 anteriores
```

**Nenhum notebook importa de outro.** Cada um é autossuficiente.
O professor pode executar em qualquer ordem.

---

## 6. Notebook 01 — Modelos LLM e NLP

**Rubrica 1 (5 itens):** demonstrar modelos pré-treinados, configurar tokenizers,
comparar arquiteturas, explicar diferenças, relacionar ao domínio.

### Células (10)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | Setup: `import torch`, `from transformers import pipeline, AutoModel, AutoTokenizer`. GPU check. |
| 3 | Markdown | **2.1 AutoModel + AutoTokenizer** — explicação do estilo do professor |
| 4 | Code | Carregar `clinicalnerpt-chemical`, tokenizar, inspecionar `last_hidden_state.shape`. Mostrar tokens WordPiece. |
| 5 | Markdown | **2.2 sentiment-analysis em frases clínicas** |
| 6 | Code | `pipeline("sentiment-analysis")` com 5 frases reais de bulas. Mostrar limitação: tom emocional vs significado clínico. |
| 7 | Markdown | **2.3 NER com clinicalnerpt-chemical** |
| 8 | Code | NER em trecho real de bula. Extrair entidades, mostrar agregação de sub-tokens. Explicar encoder-only para NER. |
| 9 | Markdown | **2.4 Tabela comparativa de arquiteturas** |
| 10 | Code + Markdown | Tabela: encoder-only (BERT), decoder-only (GPT-2), encoder-decoder (BART). Pipeline vs inferência manual. Limite 512 tokens. Conclusão: quais tarefas importam para o detector. |

### Por que 10 células e não 26?

O notebook v2.0 tinha 26 células porque cada pipeline ocupava 3 células
(markdown → code → markdown) × 6 pipelines = 18 + 8 de setup/conclusão.

No reboot, **selecionamos apenas os 3 pipelines mais relevantes** para o domínio
(NER, sentiment como demonstração de limitação, AutoModel como base) e consolidamos
a análise na tabela comparativa. Fill-mask, text-generation, summarization e QA
são **interessantes mas não essenciais** para demonstrar a rubrica 1.

**Se o professor quiser ver mais pipelines**, o notebook 05 (RAG) já demonstra
uso prático de todos os conceitos.

---

## 7. Notebook 02 — Prompt Engineering

**Rubrica 2 (5 itens):** chamadas a APIs, 3+ técnicas, prompts estruturados,
saída JSON com parsing, avaliação e iteração.

### Estratégia de dados

Em vez de anotar 1.500 pares, criamos um **ground truth sintético** de 30 pares
medicamentosos balanceados (10 de cada classe). Os textos são extraídos manualmente
de 5-6 bulas reais (Amoxicilina, Zarator, Zocor, Zyloric, Captopril) — já temos
esses textos do projeto de referência.

Isso é suficiente para demonstrar as 3 técnicas e fazer avaliação quantitativa,
sem o overhead de um pipeline de anotação.

### Células (9)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | Setup: `import openai`, `import json`, carregar 30 pares ground truth de um dict inline |
| 3 | Markdown | **Template de prompt base:** papel + tarefa + classes + contexto + formato JSON |
| 4 | Code + Markdown | **Técnica 1 — Zero-shot.** Executar nos 30 pares. Coletar acurácia, % JSON válido. |
| 5 | Code + Markdown | **Técnica 2 — Few-shot (3 exemplos).** Adicionar 1 exemplo de cada classe. Métricas. |
| 6 | Code + Markdown | **Técnica 3 — Chain-of-Thought.** Instrução "pense passo a passo". Métricas. |
| 7 | Code | `parse_interaction_response()` — tenta `json.loads()`, fallback regex. |
| 8 | Markdown | **Tabela comparativa** com acurácia, F1, JSON válido, latência. Iteração documentada. |
| 9 | Markdown | Prompt injection: demonstração de ataque + sanitização. |

---

## 8. Notebook 03 — Embeddings e Busca Vetorial

**Rubrica 3 (5 itens):** gerar embeddings, busca semântica/híbrida, avaliar modelos,
analisar acertos/falhas, justificar estratégia.

### Estratégia de dados

O notebook lê **diretamente** os arquivos `.txt` das bulas podadas (Fonte 1 e Fonte 2),
extrai as seções de interação com regex simples (mesma lógica do `preprocess.py`,
mas inline), e gera embeddings.

**Sem JSONL intermediário. Sem ChromaDB.** FAISS em memória.

### Células (8)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | Setup: `sentence-transformers`, `faiss`, `numpy`. Carregar modelo BERT pt. |
| 3 | Code | Função `carregar_chunks(dir, max_por_fonte)` — lê .txt, extrai seções, chunk em sentenças. |
| 4 | Code | Gerar embeddings para ~5.000 chunks (Fonte 2 primeiro, mais relevantes). Indexar no FAISS. |
| 5 | Code + Markdown | **Busca semântica (cosseno).** 10 queries de teste. Top-5 resultados. |
| 6 | Code + Markdown | **Busca híbrida (BM25 + embeddings).** Comparação lado a lado. |
| 7 | Code + Markdown | **Comparação de 2 modelos:** BERT pt vs MiniLM. Precision@5. |
| 8 | Markdown | Análise de 3 acertos + 3 falhas. Justificativa da estratégia (FAISS + cosseno). |

---

## 9. Notebook 04 — Inferência Local vs Remota

**Rubrica 4 (5 itens):** executar local + remoto, comparar dimensões, integrar
programaticamente, analisar trade-offs, considerar privacidade.

### Células (7)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | Setup: `openai`, `gpt4all`. Instanciar ambos. |
| 3 | Code + Markdown | **Qualidade:** classificar 30 pares com cada backend. Tabela comparativa. |
| 4 | Code + Markdown | **Latência:** 10 consultas, média/p95. |
| 5 | Markdown | **Custo:** tabela para 1K/10K/100K consultas. |
| 6 | Markdown | **Privacidade e controle:** análise LGPD, offline vs online. |
| 7 | Markdown | **Conclusão:** tabela 5 dimensões + recomendação. |

---

## 10. Notebook 05 — Pipeline RAG

**Rubrica 5 (11 itens):** pipeline completo, vector store, chunking strategies,
com/sem contexto, falhas, segurança, problema aderente, executável, integrado,
decisões justificadas, sem expor chaves, análise crítica.

Este é o notebook **mais importante** — é onde todos os conceitos se integram.

### Células (12)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica, diagrama ASCII da arquitetura |
| 2 | Code | **Setup:** carregar NER (`clinicalnerpt-chemical`), embeddings (`bert-base-portuguese-cased`), LLM (`openai`). |
| 3 | Code | **Carregar e indexar bulas:** ler Fonte 1 + Fonte 2, extrair seções de interação, chunk, gerar embeddings, FAISS. |
| 4 | Code | **Função `consultar(query)`:** NER → busca FAISS → construir prompt → LLM → parse JSON. |
| 5 | Code + Markdown | **8 consultas de demonstração** (mesmas do plano v2.0: grave, leve, sem, não encontrado, múltiplos, nome comercial, ambígua). |
| 6 | Code + Markdown | **Comparação com vs sem contexto:** 5 consultas, modo A (zero-shot) vs modo B (com chunks). Exemplo de alucinação reduzida. |
| 7 | Code + Markdown | **Estratégias de chunking:** sentenças vs parágrafos de 3 vs parágrafos de 5. Recall@5. |
| 8 | Markdown | **Análise de falhas:** 3 cenários (NER falha, chunk irrelevante, classificação errada) com causa + mitigação. |
| 9 | Code + Markdown | **Prompt injection:** demonstração de ataque + sanitização. |
| 10 | Markdown | **Riscos de segurança:** prompt injection, vazamento de contexto, data poisoning. Controles propostos. |
| 11 | Markdown | **Limitações:** NER treinado em corpus geral, cobertura das bulas, sem suporte a 3+ medicamentos. |
| 12 | Markdown | **Conclusão:** resumo do pipeline, próximos passos. |

---

## 11. README + Relatório PDF

### README.md (1 página)

-   Título e descrição (2 parágrafos)
-   Requisitos: Python 3.9+, 8 GB RAM, GPU opcional
-   Instalação: `pip install -r requirements.txt`
-   Dados: instrução para copiar de `python-processador-bulas/data/pruned/`
-   Execução: abrir notebooks na ordem c01→c05
-   Configuração: `.env.example` → `.env`

### Relatório PDF

As 26 seções obrigatórias, cada uma referenciando células específicas dos notebooks.

---

## 12. Mapeamento Rubricas → Células

Cada item de rubrica é coberto por células **numeradas** que o professor pode
verificar diretamente.

### Rubrica 1 (5 itens) → Notebook 01

| # | Item | Célula |
|---|---|---|
| 1.1 | Tarefas NLP com modelos pré-treinados | 6 (sentiment), 8 (NER) |
| 1.2 | Configurou tokenizers, pipelines, parâmetros | 4 (AutoModel), 6, 8 |
| 1.3 | Comparou modelos/arquiteturas | 9 (tabela) |
| 1.4 | Explicou diferenças (encoder-only vs decoder-only) | 9 (Markdown) |
| 1.5 | Relacionou resultados ao domínio | 10 (conclusão) |

### Rubrica 2 (5 itens) → Notebook 02

| # | Item | Célula |
|---|---|---|
| 2.1 | Chamadas a APIs/modelos | 4, 5, 6 |
| 2.2 | 3 técnicas comparadas | 4 (zero-shot), 5 (few-shot), 6 (CoT) |
| 2.3 | Prompts estruturados (papel+tarefa+formato) | 3 |
| 2.4 | JSON + parsing/validação | 7 |
| 2.5 | Avaliou e iterou prompts | 8 |

### Rubrica 3 (5 itens) → Notebook 03

| # | Item | Célula |
|---|---|---|
| 3.1 | Gerou embeddings | 4 |
| 3.2 | Busca semântica/híbrida | 5, 6 |
| 3.3 | Avaliou modelos/métricas | 7 |
| 3.4 | Analisou acertos e falhas | 8 |
| 3.5 | Justificou estratégia | 8 |

### Rubrica 4 (5 itens) → Notebook 04

| # | Item | Célula |
|---|---|---|
| 4.1 | Modelo local + remoto | 2, 3 |
| 4.2 | Comparou dimensões | 3, 4, 5, 6 |
| 4.3 | Integração programática | 2 |
| 4.4 | Vantagens/limitações | 6, 7 |
| 4.5 | Privacidade/custo/latência/controle | 5, 6, 7 |

### Rubrica 5 (11 itens) → Notebook 05 + README + Relatório

| # | Item | Célula/Arquivo |
|---|---|---|
| 5.1 | Pipeline RAG completo | 3, 4, 5 |
| 5.2 | Vector store funcional | 3 (FAISS) |
| 5.3 | Chunking/recuperação com/sem contexto | 6, 7 |
| 5.4 | Pontos de falha | 8 |
| 5.5 | Riscos de segurança | 9, 10 |
| 5.6 | Problema aderente | 1 (Markdown) |
| 5.7 | Solução executável/documentada | README.md |
| 5.8 | Integração coerente | 5 (fluxo completo) |
| 5.9 | Decisões justificadas | 7, 8, 11 |
| 5.10 | Não expôs chaves | `.env.example` + `.gitignore` |
| 5.11 | Análise crítica | 11, 12 |

---

## 13. Plano de Execução

### Pré-requisito (5 minutos)

Copiar dados do projeto de referência:

```bash
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte1 C:/workspace/python/projeto-2-modulo-1-pos/data/bulas/
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte2 C:/workspace/python/projeto-2-modulo-1-pos/data/bulas/
```

### Ordem de implementação

| # | Artefato | Tempo | Depende de |
|---|---|---|---|
| 1 | `requirements.txt`, `.env.example`, `.gitignore` | 10 min | nada |
| 2 | `c01_modelos_llm.ipynb` | 1-2 h | requisitos instalados |
| 3 | `c02_prompting.ipynb` | 1-2 h | OPENAI_API_KEY |
| 4 | `c03_embeddings_busca.ipynb` | 1-2 h | dados copiados |
| 5 | `c04_inferencia_local_ou_remota.ipynb` | 1 h | OPENAI_API_KEY |
| 6 | `c05_rag_pipeline.ipynb` | 2-3 h | todos anteriores |
| 7 | `README.md` | 30 min | projeto completo |
| 8 | Relatório PDF | 2-3 h | projeto completo |

**Total: ~12 horas de trabalho focado.**

### Commits atômicos

Um commit por notebook + um para README + um para relatório = **7 commits**.
Sem branches, sem merges, sem complexidade.

```
feat: c01_modelos_llm — 3 pipelines HuggingFace + tabela comparativa
feat: c02_prompting — 3 tecnicas de prompt + parsing JSON + validacao
feat: c03_embeddings_busca — FAISS + embeddings BERT pt + busca hibrida
feat: c04_inferencia — OpenAI vs GPT4All comparados em 5 dimensoes
feat: c05_rag_pipeline — NER + busca + LLM + analise de seguranca
docs: README + requisitos + instrucoes de reproducao
docs: relatorio PDF com 26 secoes obrigatorias
```

---

## Apêndice: Comparação Antes vs Depois

| Métrica | v2.0 (atual) | v3.0 (reboot) |
|---|---|---|
| Arquivos Python | 8 | 0 (tudo em notebooks) |
| Arquivos de doc | 13 | 1 |
| Linhas de script | ~1.500 | 0 |
| Células de notebook | 26 (c01) + 0 (c02-c05) | 10 + 9 + 8 + 7 + 12 = 46 |
| Dependências | 15 | 8 |
| Dados intermediários | JSONL 270k linhas, 4 CSVs | Leitura direta dos .txt |
| Vector store | ChromaDB (persistente) | FAISS (em memória) |
| Tempo estimado | ~21 dias | ~3 dias |
| Commits | 13 (e subindo) | 7 |
| Complexidade ciclomática | Alta (scripts interdependentes) | Baixa (notebooks isolados) |
| Cobre 30 rubricas? | Sim (teoricamente) | Sim (com mapeamento explícito) |
