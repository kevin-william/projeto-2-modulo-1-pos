# Plano de Implementação — Detector de Interações Medicamentosas com LLMs e RAG

**Versão:** 2.0 (9 fases — inclui fine-tuning)  
**Projeto:** Módulo 1 — Sistemas Cognitivos com Large Language Models (Pós-Graduação)  
**Aluno:** Kevin Rodrigues  
**Total de bulas:** 5.960 (Fonte 1 / ANVISA: 4.978 | Fonte 2 / Consultaremedios: 982)

---

## Índice

1. [Visão Geral e Objetivo](#1-visão-geral-e-objetivo)
2. [Arquitetura da Solução](#2-arquitetura-da-solução)
3. [Ambiente GPU](#3-ambiente-gpu)
4. [Mapeamento Rubricas → Fases](#4-mapeamento-rubricas--fases)
5. [Estrutura do Projeto](#5-estrutura-do-projeto)
6. [Resumo das 9 Fases](#6-resumo-das-9-fases)
7. [Cronograma](#7-cronograma)
8. [Riscos e Mitigações](#8-riscos-e-mitigações)

---

## 1. Visão Geral e Objetivo

### Problema

Profissionais de saúde e pacientes precisam verificar interações entre medicamentos antes da administração concomitante. As bulas oficiais (ANVISA) e bases como Consultaremedios contêm essas informações, mas em formato textual extenso (até 10 mil tokens por bula), inviabilizando consulta rápida. Este projeto constrói um sistema cognitivo que, dada uma consulta em linguagem natural como _"Posso tomar Amoxicilina com Ibuprofeno?"_, recupera os trechos relevantes das bulas, classifica a interação e gera uma resposta fundamentada.

### Dataset

| Característica | Fonte 1 (ANVISA) | Fonte 2 (Consultaremedios) |
|---|---|---|
| Quantidade | 4.978 | 982 |
| Formato | Texto corrido, seções por cabeçalhos | 16 blocos Q&A estruturados |
| Versões | `_paciente` e `_profissional` | Somente `_profissional` |
| Seções-chave | "Interações Medicamentosas", "Precauções", "Contraindicações" | Bloco 10: `INTERAÇÃO MEDICAMENTOSA?` |
| Desafio | Texto longo, precisa de seccionamento seletivo | Respostas possivelmente truncadas |

### Por que LLMs?

- **NER especializado**: Extrair entidades químicas de texto não estruturado — impossível com regex puro.
- **Classificação contextual**: Distinguir entre "não há interação" e "interação grave" exige compreensão semântica.
- **RAG**: Recuperar os trechos exatos que fundamentam a classificação, permitindo auditoria e reduzindo alucinação.

---

## 2. Arquitetura da Solução

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CONSULTA DO USUÁRIO (NL)                          │
│           "Posso tomar Amoxicilina com Ibuprofeno?"                 │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FASE 1-2: PRÉ-PROCESSAMENTO (executado offline, uma vez)            │
│                                                                       │
│  5.960 .txt ──► Parse seletivo ──► Chunking ──► Indexação ChromaDB   │
│                 (seções F1 / Q10 F2)    (sentenças)    + embeddings   │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FASE 3-4: ANOTAÇÃO + FINE-TUNING (executado offline, uma vez)       │
│                                                                       │
│  Weak supervision ──► 1.500 pares ──► BioBERTpt fine-tuned            │
│  (Fonte 2 heurística)   rotulados      (3 classes, F1 ≥ 0.80)        │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FASE 8: PIPELINE RAG (online, por consulta)                          │
│                                                                       │
│  ┌──────────┐   ┌───────────────┐   ┌──────────────────┐             │
│  │ NER GPU  │   │ Busca Vetorial │   │ Classificador     │             │
│  │(clinical │   │ (ChromaDB +    │   │ (BioBERTpt        │             │
│  │ nerpt-   │   │  BERT pt)      │   │  fine-tuned GPU)  │             │
│  │ chemical)│   │                │   │                   │             │
│  └────┬─────┘   └───────┬────────┘   └────────┬──────────┘             │
│       │                 │                      │                       │
│       └─────────────────┴──────────────────────┘                       │
│                         │                                              │
│                         ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  LLM RAG (GPT4All local ou OpenAI remota)                       │ │
│  │  Prompt aumentado com chunks recuperados + classificações       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                         │                                              │
│                         ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  SAÍDA JSON:                                                     │ │
│  │  {                                                               │ │
│  │    "consulta": "Amoxicilina + Ibuprofeno",                       │ │
│  │    "interacoes": [{                                               │ │
│  │      "medicamento": "Ibuprofeno",                                 │ │
│  │      "classe": "LEVE_MODERADA",                                   │ │
│  │      "confianca": 0.87,                                           │ │
│  │      "evidencia": "...",                                          │ │
│  │      "fonte": "bula_amoxicilina_profissional.txt"                 │ │
│  │    }]                                                             │ │
│  │  }                                                               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Stack Tecnológica

| Camada | Tecnologia | Por quê | Dispositivo |
|---|---|---|---|
| **NER** | `pucpr/clinicalnerpt-chemical` | Treinado em nomes comerciais + princípios ativos | **GPU** (440 MB) |
| **Classificador** | `pucpr/biobertpt-all` **fine-tuned** | ~110M parâmetros, 3 classes, F1 ≥ 0.80 | **GPU** (442 MB) |
| **Embeddings** | `neuralmind/bert-base-portuguese-cased` | BERT pt consolidado, 768 dims, compatível ChromaDB | GPU sob demanda |
| **Vector Store** | ChromaDB (persistente) | Sem servidor externo, API Python nativa | CPU/disco |
| **LLM RAG** | GPT4All (Phi-3-mini Q4_K_M) + OpenAI (GPT-4o-mini) | Local = privacidade; Remoto = qualidade | CPU (GGUF) |
| **Interface** | Jupyter Notebooks (.ipynb) + scripts Python | Formato exigido pelo professor | — |

---

## 3. Ambiente GPU

| Componente | Valor |
|---|---|
| **GPU** | NVIDIA GeForce RTX 3050 6GB Laptop |
| **Compute Capability** | 8.6 |
| **Driver** | 591.59 |
| **CUDA Toolkit** | 12.4 (recomendado para PyTorch 2.5.1) |
| **PyTorch atual** | ❌ 2.3.1+cpu (precisa reinstalar) |

### Orçamento de VRAM

| Modelo | VRAM (FP16) | Simultâneo? |
|---|---|---|
| `clinicalnerpt-chemical` (BERT 110M) | ~440 MB | ✅ |
| `biobertpt-all` (BERT 110M) | ~440 MB | ✅ |
| `biobertpt-all` fine-tuned (classifier head) | ~442 MB | ✅ |
| Overhead CUDA + PyTorch | ~500 MB | |
| **Pico estimado** | **~1.8 GB** | ✅ 30% da VRAM |

**Ação crítica (Fase 0):** Criar venv e instalar PyTorch com suporte CUDA:
```bash
cd C:\workspace\python\projeto-2-modulo-1-pos
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash / MSYS)
# ou: venv\Scripts\activate    # Windows (CMD)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. Mapeamento Rubricas → Fases

| Rubrica (30 itens) | Fase | Entregável |
|---|---|---|
| **1. NLP + Hugging Face** (5 itens) | Fase 2 | `c01_modelos_llm.ipynb` |
| **2. Prompt Engineering** (5 itens) | Fase 5 | `c02_prompting.ipynb` |
| **3. Embeddings + Busca** (5 itens) | Fase 6 | `c03_embeddings_busca.ipynb` |
| **4. Inferência Local vs Remota** (5 itens) | Fase 7 | `c04_inferencia_local_ou_remota.ipynb` |
| **5. Pipeline RAG** (11 itens) | Fases 8–9 | `c05_rag_pipeline.ipynb` + Relatório |
| **Fine-tuning** (extra, fortalece Rubrica 1) | Fases 3–4 | `scripts/classifier.py` + modelo fine-tuned |

---

## 5. Estrutura do Projeto

```
C:\workspace\python\projeto-2-modulo-1-pos\
├── c01_modelos_llm.ipynb
├── c02_prompting.ipynb
├── c03_embeddings_busca.ipynb
├── c04_inferencia_local_ou_remota.ipynb
├── c05_rag_pipeline.ipynb
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── venv/                             ← Ambiente virtual (não versionado)
├── docs/
│   ├── PLANO_IMPLEMENTACAO.md          ← Este arquivo (v2.0)
│   ├── dataset/
│   │   └── RESUMO_DATASET_BULAS.md
│   └── fases/
│       ├── FASE_0.md                   ← Estrutura + Setup GPU
│       ├── FASE_1.md                   ← Pré-processamento das bulas
│       ├── FASE_2.md                   ← Notebook 01: HF + NLP
│       ├── FASE_3.md                   ← Anotação do dataset
│       ├── FASE_4.md                   ← Fine-tuning BioBERTpt
│       ├── FASE_5.md                   ← Notebook 02: Prompt Engineering
│       ├── FASE_6.md                   ← Notebook 03: Embeddings + ChromaDB
│       ├── FASE_7.md                   ← Notebook 04: Inferência
│       ├── FASE_8.md                   ← Notebook 05: RAG Pipeline
│       └── FASE_9.md                   ← Relatório PDF + README
├── scripts/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── annotate.py                     ← Weak supervision + curadoria
│   ├── train_classifier.py             ← Fine-tuning BioBERTpt
│   ├── ner.py
│   ├── classifier.py
│   ├── embeddings.py
│   ├── rag.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_preprocess.py
│   ├── test_annotate.py
│   ├── test_train_classifier.py
│   ├── test_ner.py
│   ├── test_classifier.py
│   ├── test_embeddings.py
│   └── test_rag.py
├── data/
│   ├── bulas/
│   │   ├── fonte1/                     # 4.978 .txt (não versionado)
│   │   └── fonte2/                     # 982 .txt (não versionado)
│   ├── chroma_db/                      # Persistência ChromaDB
│   ├── anotacoes/
│   │   ├── automaticas.csv             # ~1.000 pares (weak supervision)
│   │   └── manuais.csv                 # ~500 pares (curadoria)
│   └── modelos_finetuned/
│       └── biobertpt-interactions/     # Modelo fine-tuned salvo
└── logs/
    └── pipeline.log
```

---

## 6. Resumo das 9 Fases

| Fase | Nome | Dias | Entregável principal | Rubricas |
|---|---|---|---|---|
| **0** | Estrutura + Setup GPU | 1 | `config.py`, `requirements.txt`, PyTorch CUDA | — |
| **1** | Pré-processamento das Bulas | 2 | `scripts/preprocess.py`, chunks JSONL | — |
| **2** | Notebook 01 — HF + NLP | 2 | `c01_modelos_llm.ipynb` | Rubrica 1 (5/5) |
| **3** | Anotação do Dataset | 3 | `data/anotacoes/` (~1.500 pares) | — |
| **4** | Fine-Tuning BioBERTpt | 3 | Modelo fine-tuned, `scripts/classifier.py` | Rubrica 1 (extra) |
| **5** | Notebook 02 — Prompt Engineering | 2 | `c02_prompting.ipynb` | Rubrica 2 (5/5) |
| **6** | Notebook 03 — Embeddings + ChromaDB | 2 | `c03_embeddings_busca.ipynb` | Rubrica 3 (5/5) |
| **7** | Notebook 04 — Inferência | 1 | `c04_inferencia_local_ou_remota.ipynb` | Rubrica 4 (5/5) |
| **8** | Notebook 05 — RAG Pipeline | 3 | `c05_rag_pipeline.ipynb`, `scripts/rag.py` | Rubrica 5 (11/11) |
| **9** | Relatório PDF + README | 2 | `README.md`, PDF | Rubrica 5 (docs) |

**Total: ~21 dias úteis (4 semanas)**

---

## 7. Cronograma

| Semana | Dias | Fases | Entregáveis |
|---|---|---|---|
| **1** | 1–5 | 0, 1, 2 | Estrutura, pré-processamento, Notebook 01 |
| **2** | 6–10 | 3, 4 | Anotação, fine-tuning BioBERTpt |
| **3** | 11–15 | 5, 6, 7 | Notebooks 02, 03, 04 |
| **4** | 16–21 | 8, 9 | Notebook 05 (RAG), relatório, README |

---

## 8. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| PyTorch CUDA não instalar no Windows + Python 3.12 | Média | Alto | Fallback: CPU para tudo (mais lento, mas funcional); comprovar com `torch.cuda.is_available()` |
| VRAM insuficiente para NER + classificador simultâneos | Baixa | Médio | Carregar sequencialmente com `torch.cuda.empty_cache()` entre etapas |
| Dados anotados insuficientes para fine-tuning (< 500 pares) | Média | Alto | Aumentar weak supervision da Fonte 2; reduzir para classificação binária se necessário |
| Overfitting no fine-tuning (dataset pequeno) | Média | Médio | Early stopping, weight decay, dropout 0.3, gradient accumulation |
| APIs OpenAI com custo inesperado | Baixa | Baixo | GPT-4o-mini (mais barato); demonstrar com ~50 chamadas; fallback GPT4All |
| Professor não conseguir reproduzir | Média | Alto | `requirements.txt` com versões exatas `==`; testar em venv limpo antes da entrega |
| Bulas ausentes no caminho esperado | Baixa | Médio | Instruções claras no README; script de verificação de integridade |

---

## Apêndice: Checklist de Rubricas (30 itens)

### Rubrica 1: NLP com LLMs e Hugging Face (5 itens)
- [ ] Tarefas NLP com modelos pré-treinados → Fase 2, Notebook 01
- [ ] Configurou modelos, tokenizers, pipelines → Fase 2
- [ ] Comparou modelos/arquiteturas → Fase 2, tabela comparativa
- [ ] Explicou diferenças entre tipos de modelo → Fase 2, células Markdown
- [ ] Relacionou resultados ao caso de uso → Fase 2, conclusão

### Rubrica 2: Prompt Engineering (5 itens)
- [ ] Chamadas a APIs/modelos → Fase 5, Notebook 02
- [ ] Comparou técnicas de prompting → Fase 5 (zero-shot, few-shot, CoT)
- [ ] Estruturou prompts → Fase 5, template [PAPEL+TAREFA+CONTEXTO+JSON]
- [ ] Saídas JSON + parsing → Fase 5, `parse_interaction_response()`
- [ ] Avaliou e iterou prompts → Fase 5, 200 pares, 3 iterações

### Rubrica 3: Embeddings e Busca Vetorial (5 itens)
- [ ] Gerou embeddings → Fase 6, Notebook 03
- [ ] Busca semântica/híbrida → Fase 6, cosseno + BM25
- [ ] Avaliou modelos/métricas → Fase 6, 3 modelos comparados
- [ ] Analisou acertos e falhas → Fase 6, 5+5 casos
- [ ] Justificou estratégia → Fase 6, célula Markdown

### Rubrica 4: Inferência Local vs Remota (5 itens)
- [ ] Modelo local + remoto → Fase 7, Notebook 04
- [ ] Comparou requisitos/desempenho → Fase 7, tabela 5 dimensões
- [ ] Integração programática → Fase 7, classe `LLMProvider`
- [ ] Vantagens/limitações → Fase 7, análise
- [ ] Privacidade/custo/latência/controle → Fase 7, seção dedicada

### Rubrica 5: Pipeline RAG (11 itens)
- [ ] Pipeline RAG completo → Fase 8, Notebook 05
- [ ] Vector store funcional → Fase 8, ChromaDB
- [ ] Chunking/recuperação com/sem contexto → Fase 8
- [ ] Pontos de falha → Fase 8
- [ ] Riscos de segurança → Fase 8
- [ ] Problema aderente → Introdução de cada notebook
- [ ] Solução executável/documentada → Fase 9, README
- [ ] Integrou LLMs, prompts, embeddings, busca → Fase 8
- [ ] Justificou decisões com resultados → Todas as fases
- [ ] Não expôs chaves/tokens → `.env.example`, `.gitignore`
- [ ] Análise crítica de limitações → Fase 9, Relatório
