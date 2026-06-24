# Arquitetura do Projeto — Detector de Interações Medicamentosas

## Visão Geral

Pipeline end-to-end que, dado um par de medicamentos e um contexto clínico, classifica se há interação (0=nenhuma, 1=moderada, 2=grave) e justifica a decisão com base nas bulas.

O sistema funciona em duas modalidades complementares:

- **RAG (Retrieval-Augmented Generation)**: dado um par de drogas e uma pergunta, recupera trechos relevantes das bulas e usa LLM para classificar e justificar.
- **Classificador fine-tuned**: modelo BioBERTpt treinado diretamente nos dados anotados para classificação em 3 classes.

---

## Stack Tecnológico

| Camada | Tecnologia | Para que serve |
|--------|------------|---------------|
| **LLM** | DeepSeek API (remoto) / GPT4All (local) | Geração de texto, classificação com justificativa |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` (384d) | Transformar texto em vetores numéricos para busca |
| **Vetorstore** | ChromaDB | Armazenar e buscar chunks de bulas por similaridade |
| **Classificador** | `pucpr/biobertpt-all` fine-tuned (3 classes) | Classificar interações diretamente (sem LLM) |
| **NER** | `pucpr/clinicalnerpt-chemical` | Extrair nomes de medicamentos de texto livre |
| **Dados** | Bulário ANVISA (fonte1) + Consultaremedios (fonte2) | Fonte de verdade para interações medicamentosas |

---

## Estrutura de Dados

### `data/bulas/`

As bulas são a matéria-prima do sistema. Existem duas fontes, com formatos completamente diferentes:

```
bulas/
  fonte1/   ← Bulário ANVISA (~500 arquivos .txt)
  fonte2/   ← Consultaremedios (~982 arquivos .txt)
```

**Fonte 1 (ANVISA):** Cada arquivo contém uma bula completa de um medicamento, separada em duas versões — paciente (linguagem simples) e profissional (linguagem técnica). Os arquivos seguem um padrão de nomenclatura como `100290226_etoricoxibe_paciente.txt`, onde o número inicial é o registro ANVISA.

**Fonte 2 (Consultaremedios):** Arquivos menores, no formato de perguntas e respostas (Q&A). Cada bloco começa com `[P:]` (pergunta) e `[R:]` (resposta). Contêm informações sobre interações, indicações, efeitos colaterais, etc.

As bulas não são usadas em texto corrido — passam por um pré-processamento que extrai apenas as seções relevantes (interações, contraindicações, advertências) antes de serem chunkadas e indexadas.

### `data/anotacoes/`

Este diretório contém os datasets anotados que alimentam o treinamento do classificador.

| Arquivo | Conteúdo | Uso |
|---------|----------|-----|
| `manuais.csv` | Anotações manuais originais — 239 contextos únicos, 1060 pares droga+contexto+classe | Base do dataset |
| `automaticas.csv` | Anotações geradas automaticamente por LLM | Ampliar dataset |
| `train.csv` | 80% dos contextos — 850 pares, 190 ctx únicos | Treino do classificador |
| `val.csv` | 10% — 105 pares, 24 ctx únicos | Validação durante treino |
| `test.csv` | 10% — 105 pares, 24 ctx únicos | Avaliação final |
| `pendentes_curadoria.csv` | Casos que ainda precisam de revisão humana | Trabalho de curadoria |

> **Sobre o split**: foi usado GroupKFold usando `contexto` como grupo. Isso significa que cada contexto clínico (ex: "paciente renal", "gestante", "usuário de anticoagulante") aparece inteiro em apenas um dos três splits. O modelo não consegue decorar "contexto X → classe Y" porque nunca vê o mesmo contexto em treino e teste. O split anterior tinha um vazamento grave: 99% dos contextos de teste apareciam no treino, fazendo o modelo Performar 65% só por memorização.

Cada linha do CSV segue o formato: `medicamento_alvo`, `medicamento_outro`, `contexto`, `classe`, `justificativa`.

- **classe 0** = nenhuma interação
- **classe 1** = interação moderada
- **classe 2** = interação grave

### `data/chunks_bulas.jsonl`

Bulas pré-processadas e divididas em pedaços menores (chunks). Cada chunk é uma frase ou bloco de texto de no máximo 512 tokens. O arquivo tem ~156.000 linhas, cada uma com:

```json
{"id": "cvar_087", "texto": "A associacao de sinvastatina com itraconazol e contraindicada...", "medicamento": "sinvastatina", "fonte": "fonte1"}
```

Esse arquivo é a entrada para a indexação no ChromaDB. Os chunks menores garantem que a busca vetorial retorne trechos específicos e relevantes, não páginas inteiras de bula.

### `data/modelos_finetuned/biobertpt-interactions/`

Modelo BioBERTpt fine-tuned com os pesos finais do treinamento. Contém:

- `config.json` — configuração do modelo
- `model.safetensors` — pesos do modelo (~500MB)
- `tokenizer/` — tokenizer treinado no vocabulário do BioBERTpt
- `tokenizer_config.json` — configurações do tokenizer
- `trainer_state.json` — histórico de métricas do treinamento

Esse é o modelo que classifica pares droga+droga+contexto sem precisar de LLM externo.

---

## Scripts (`scripts/`)

### `config.py`

Arquivo central de configurações. Todas as constantes compartilhadas ficam aqui — caminhos de dados, nomes de modelos, padrões regex para parsing de bulas, hyperparameters, etc. qualquer script que precise de uma configuração vai importar daqui. Se precisares mudar o modelo de embedding ou o diretório de dados, mexe só aqui.

### `preprocess.py`

Responsável por transformar os arquivos `.txt` crus das bulas em `chunks_bulas.jsonl`. É a esteira de preparação de dados.

**O que faz, passo a passo:**

1. **Lê os arquivos da fonte1** — usa regex para detectar seções relevantes (que contenham "INTERACAO", "CONTRAINDIC", "ADVERTENCIA", etc.) e descarta o resto
2. **Lê os arquivos da fonte2** — extrai blocos Q&A que tratem de interações medicamentosas, identificados pela marcação `[P:]` / `[R:]`
3. **Normaliza texto** — remove acentuação excessiva, padroniza whitespace, converte para minúsculas onde apropriado
4. **Divide em chunks** — cada chunk tem no máximo 512 tokens (usando 4 caracteres por token como aproximação)
5. **Gera IDs únicos** — cada chunk recebe um ID no formato `medicamento_numero` (ex: `sinv_001`)
6. **Salva em JSONL** — um chunk por linha, pronto para indexação

| Entrada | Saída |
|---------|-------|
| `data/bulas/fonte1/` + `data/bulas/fonte2/` | `data/chunks_bulas.jsonl` (~156K chunks) |

Executar: `python scripts/preprocess.py --input data/bulas --output data/chunks_bulas.jsonl`

### `embeddings.py`

Responsável por tudo relacionado a vetorizar textos e buscar no ChromaDB.

**Funções disponíveis:**

| Função | O que faz |
|--------|-----------|
| `gerar_embeddings(textos, modelo_nome)` | Recebe uma lista de textos e retorna uma array numpy de vetores 384d usando SentenceTransformer |
| `criar_collection(recreate)` | Cria uma nova collection no ChromaDB. Se `recreate=True`, apaga a existente primeiro |
| `indexar_chunks(collection, df, embeddings)` | Insere os chunks em lotes de 100 no ChromaDB, junto com metadados (medicamento, fonte) |
| `construir_index(chunks_path, modelo, recreate)` | Orquestra o pipeline completo: carregar JSONL → gerar embeddings → indexar. Se `recreate=False`, reaproveita se já existir |
| `buscar_chunks(collection, embedding, n)` | Recebe um vetor de query, busca os n chunks mais próximos por similaridade de cosseno no ChromaDB, retorna lista de dicts com id, texto, medicamento, distância |
| `busca_hibrida(collection, query, modelo, n, alpha)` | Combina busca vetorial (alpha) com busca por palavra-chave (1-alpha). Útil quando o modelo vetorial erra por phrasing diferente |

**Modelo usado**: `sentence-transformers/all-MiniLM-L6-v2` — leve (384d), rápido, bom para português. Foi preferido sobre o BERTPT porque modelos de sentence embeddings são otimizados para similaridade semântica, enquanto BERTPT é MLM genérico.

**Fluxo típico de uso:**

```python
from scripts.embeddings import construir_index

# Primeira vez — cria o índice
collection = construir_index(
    chunks_path="data/chunks_bulas.jsonl",
    modelo_embedding="sentence-transformers/all-MiniLM-L6-v2",
    recreate=True
)

# Depois — reaproveita o índice existente
collection = construir_index(
    chunks_path="data/chunks_bulas.jsonl",
    modelo_embedding="sentence-transformers/all-MiniLM-L6-v2",
    recreate=False  # muito mais rápido
)
```

### `ner.py` — `MedicationNER`

Extrai nomes de medicamentos de texto não estruturado. Usa o modelo `clinicalnerpt-chemical` treinado especificamente para reconhecer entidades químicas e medicamentos em textos médicos em português.

**Características:**
- Agrega sub-tokens (B-ChemicalDrugs + I-ChemicalDrugs → nome completo)
- Normaliza variações do mesmo medicamento (ex: "Metformina", "metformina", "cloridrato de metformina" → normalizado)
- Remove duplicatas

```python
from scripts.ner import MedicationNER
ner = MedicationNER()

drogas = ner.extract("Paciente de 65 anos usando metformina para diabetes e sinvastatina para colesterol")
# ['metformina', 'sinvastatina']

drogas = ner.extract("associacao de captopril com diurético")
# ['captopril', 'diurético']
```

Se a extração falha ou retorna menos de 2 drogas, o pipeline RAG自知 sabe que precisa pedir clarification ao usuário.

### `classifier.py` — `InteractionClassifier`

Wrapper de inferência para o modelo BioBERTpt fine-tuned. Carrega os pesos de `data/modelos_finetuned/biobertpt-interactions/` e oferece interface simples para classificação.

```python
from scripts.classifier import InteractionClassifier
clf = InteractionClassifier()

# Classificar um par individual
result = clf.classificar("amoxicilina", "ibuprofeno", "paciente com infecção e dor")
# {
#   "classe": 1,
#   "confianca": 0.87,
#   "probabilidades": [0.05, 0.87, 0.08]
# }

# Classificar múltiplos pares de uma vez
results = clf.classificar_lote([
    ("amoxicilina", "ibuprofeno", "infecção e dor"),
    ("varfarina", "aspirina", "paciente com trombose"),
])
```

### `train_classifier.py`

Script de treinamento do classificador. Faz fine-tuning do `pucpr/biobertpt-all` para as 3 classes de interação.

**Arquitetura do modelo:**
```
[CLS] medicamento_alvo [SEP] medicamento_outro [SEP] contexto [SEP]
    → BioBERTpt (768d de saída do [CLS])
    → Dropout(0.3)
    → Linear(768 → 3)
    → Softmax → probabilities
```

**Hiperparâmetros:**
- Batch size: 16
- Gradient accumulation: 4 (effective batch = 64)
- Learning rate: 2e-5 com warmup
- Early stopping: monitora val_loss, patience=2
- Mixed precision: FP16
- Class weights: [1.0, 2.0, 3.0] (classe 2 — interação grave — tem peso maior porque é mais rara)

Executar: `python scripts/train_classifier.py`

### `rag.py` — `RAGPipeline`

Orquestra todo o fluxo end-to-end de consulta. É o "cérebro" que coordena NER, busca vetorial, classificador e LLM.

**Passo a passo:**

1. **NER (step 1)** — Recebe a query do usuário, extrai os medicamentos mencionados usando `MedicationNER`
2. **Busca vetorial (step 2)** — Para cada medicamento, busca os chunks mais similares no ChromaDB. Monta uma lista de trechos relevantes das bulas
3. **Classificador (step 3)** — Usa o BioBERTpt fine-tuned para dar uma primeira classificação (opcional — pode ser usada só como pista para o LLM)
4. **LLM (step 4)** — Monta o prompt com: contexto clínico, medicamentos, chunks recuperados e instrução para classificar + justificar. Envia para DeepSeek (remoto) ou GPT4All (local)
5. **Retorno** — Recebe JSON estruturado com classe predita, justificativa e nível de confiança

```python
from scripts.rag import RAGPipeline

pipeline = RAGPipeline(
    vectorstore="local",    # usa ChromaDB local
    llm="deepseek"          # ou "gpt4all" para modo local
)

resultado = pipeline.consultar(
    query="Paciente renal usando metformina e anti-inflamatório",
    contexto="paciente com insuficiência renal crônica"
)
# {
#   "interacao": True,
#   "classe": 2,
#   "droga_alvo": "metformina",
#   "droga_outro": "ibuprofeno",
#   "justificativa": "Anti-inflamatórios podem pi...",
#   "fonte": "bula_relevante_id",
#   "confianca": 0.91
# }
```

---

## Notebooks

### `c01_modelos_llm.ipynb`

Notebook de **exploração e descoberta**. Antes de escolher qualquer modelo, a equipe testou vários modelos de HuggingFace disponíveis para entender o que funcionava melhor para cada tarefa do pipeline.

**O que cada célula testa:**
- **NER** com `pucpr/clinicalnerpt-chemical` — extração de medicamentos de textos médicos
- **Sumarização** com `facebook/bart-large-cnn` — resumir bulas longas
- **QA (Question Answering)** com `pierreguillou/bert-base-cased-squad-v1.1-portuguese` — encontrar respostas dentro das bulas
- **Geração de texto** com `pierreguillou/gpt2-small-portuguese` — geração livre
- **Fill-mask** com `sentence-transformers/all-MiniLM-L6-v2` — verificar embeddings

**Dependências**: nenhum arquivo local — tudo roda online via HuggingFace.

### `c02_prompting.ipynb`

Notebook de **experimentação de prompting**. Antes de mandar para o LLM em produção, a equipe comparou sistematicamente 3 estratégias de prompting para ver qual gerava melhores classificações.

**Três técnicas testadas:**

1. **Zero-shot** — Só o template com o papel definido. Ex: "Você é um farmacêutico. Classifique: [DROGA_A] + [DROGA_B]..."

2. **Few-shot** — Inclui 3 exemplos já classificados no prompt (1 de cada classe). O modelo entende o padrão de resposta esperada.

3. **CoT (Chain-of-Thought)** — Adiciona a instrução "Pense passo a passo antes de responder" para forçar raciocínio explícito.

**Avaliação**: roda as 3 técnicas nos mesmos 200 pares e compara qualidade das respostas, tempo de resposta e custo.

**Dependências**: `data/anotacoes/manuais.csv` — nenhum modelo baixado.

### `c03_embeddings_busca.ipynb`

O **centro da infraestrutura de busca**. Este notebook é onde toda a indexação vetorial acontece e onde a qualidade da busca é validada.

**Células na ordem:**

1. **Imports e config**
2. **Pré-processamento** — executa `preprocess.py` para gerar `chunks_bulas.jsonl` se ainda não existir
3. **Geração de embeddings** — vetoriza todos os chunks com MINILM (384d, ~156K chunks)
4. **Indexação no ChromaDB** — cria a collection e insere todos os chunks com metadados
5. **Testes de busca** — consultas médicas reais para verificar se os chunks certos aparecem no topo
6. **Comparação de alpha** — testa diferentes valores de alpha na busca híbrida para encontrar o melhor equilíbrio vetorial × keyword
7. **Métricas** — P@3 e MRR em 10 consultas de teste

**Dependências**: `scripts/embeddings.py`, `data/chunks_bulas.jsonl`

**Saída**: ChromaDB persistido em `data/chroma_db/` (~155K vetores × 384d)

### `c04_inferencia_local_ou_remota.ipynb`

Notebook de **decisão de infraestrutura**. O projeto pode usar LLM de duas formas — qual vale mais a pena?

**Opção 1 — Remote (DeepSeek API):**
- Prós: modelo grande e poderoso, sem consumo de GPU local, manutenção zero
- Contras: dados saem da máquina (questões de privacidade), custo por token, dependência de internet

**Opção 2 — Local (GPT4All + modelo quantizado):**
- Prós: 100% offline, privacidade total, custo só de eletricidade
- Contras: precisa de GPU com VRAM suficiente, modelos quantizados perdem qualidade

**O notebook compara**: latência (ms por query), qualidade das respostas (avaliação subjetiva), custo operacional e implicações de privacidade.

**Dependências**: nenhum arquivo local

### `c05_rag_pipeline.ipynb`

O **notebook de integração final**. Aqui o pipeline completo é montado e testado, combinando NER + ChromaDB + classificador + LLM.

**Células:**

1. **Imports** — carrega todos os módulos (`rag.py`, `embeddings.py`, `ner.py`)
2. **Demonstração RAG** — 5 consultas médicas com resposta completa (classe + justificativa + fontes)
3. **Comparação LLM puro vs LLM+RAG** — mostra a diferença de qualidade quando o LLM tem contexto das bulas vs quando não tem
4. **Análise de chunking** — testa 3 estratégias diferentes de dividir as bulas em chunks (por sentença, por parágrafo, por seção)
5. **Análise de falhas** — identifica casos onde o pipeline dá resposta ruim e explica por quê

**Dependências**: `scripts/rag.py`, `scripts/embeddings.py`, collection ChromaDB (`data/chroma_db/`), modelo fine-tuned (`data/modelos_finetuned/biobertpt-interactions/`)

---

## Fluxo de Execução

O projeto é dividido em **4 fases sequenciais** — cada uma precisa da anterior, mas roda independentemente depois de pronta:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PRE-PROCESSAMENTO (uma vez)                              │
│    python scripts/preprocess.py                             │
│    Bulário ANVISA + Consultaremedios                        │
│    → chunks_bulas.jsonl (156K chunks semânticos)            │
└────────────────────────┬────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. INDEXAÇÃO (uma vez)                                      │
│    python -c "from scripts.embeddings import construir..."  │
│    ou rode o c03_embeddings_busca.ipynb                     │
│    chunks_bulas.jsonl                                        │
│    → ChromaDB em data/chroma_db/ (155987 vetores × 384d)    │
└────────────────────────┬────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TREINAMENTO (uma vez)                                    │
│    python scripts/train_classifier.py                       │
│    train.csv + val.csv + test.csv                           │
│    → data/modelos_finetuned/biobertpt-interactions/         │
└────────────────────────┬────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. INFERÊNCIA (executa query a query)                        │
│                                                             │
│  Query ──→ NER ──→ medicamentos                             │
│              │                                              │
│              ▼                                              │
│         ChromaDB ──→ chunks relevantes                      │
│              │                                              │
│              ▼                                              │
│         Monta contexto RAG                                   │
│              │                                              │
│              ▼                                              │
│         LLM (DeepSeek ou GPT4All) ──→ JSON final            │
│         {classe, justificativa, fontes, confiança}          │
└─────────────────────────────────────────────────────────────┘
```

**Fases 1 e 2 precisam ser executadas uma única vez.** Depois de ter `chunks_bulas.jsonl` e o ChromaDB indexado, eles são reutilizados em todas as consultas.

**Fase 3 também é uma vez.** O modelo fine-tuned é treinado e salvo — depois só é carregado para inferência.

**Fase 4 é o modo de uso.** Cada nova consulta de interação medicamentosa passa por esse fluxo.

---

## Dependências Entre Arquivos

Quem lê o quê, e quem escreve o quê:

| Arquivo | Lê | Escreve |
|---------|----|---------|
| `scripts/preprocess.py` | `data/bulas/fonte1/`, `data/bulas/fonte2/` | `data/chunks_bulas.jsonl` |
| `scripts/embeddings.py` | `data/chunks_bulas.jsonl` | `data/chroma_db/` (ChromaDB persistente) |
| `scripts/train_classifier.py` | `data/anotacoes/train.csv`, `val.csv`, `test.csv` | `data/modelos_finetuned/biobertpt-interactions/` |
| `scripts/classifier.py` | `data/modelos_finetuned/biobertpt-interactions/` | — (só lê, inferência) |
| `scripts/rag.py` | `data/chroma_db/`, `data/modelos_finetuned/`, `scripts/embeddings.py` | — |
| `c01_modelos_llm.ipynb` | Modelos HuggingFace pela API | — |
| `c02_prompting.ipynb` | `data/anotacoes/manuais.csv` | — |
| `c03_embeddings_busca.ipynb` | `scripts/embeddings.py`, `data/chunks_bulas.jsonl` | `data/chroma_db/` |
| `c04_inferencia_local_ou_remota.ipynb` | nenhum arquivo local | — |
| `c05_rag_pipeline.ipynb` | `scripts/rag.py`, `scripts/embeddings.py`, ChromaDB, modelo fine-tuned | — |

---

## Fluxo de Dados Completo

```
FONTE1 — Bulário ANVISA (.txt)
  ~500 arquivos: {registro}_{medicamento}_{tipo}.txt
  Exemplo: 100290226_etoricoxibe_paciente.txt

FONTE2 — Consultaremedios (.txt)
  ~982 arquivos: {medicamento}.txt
  Formato Q&A: [P:] pergunta / [R:] resposta

  ┌─────────────────────────────────────────┐
  │ scripts/preprocess.py                   │
  │                                         │
  │ • Parseia seletivamente só seções       │
  │   relevantes (INTERACAO, CONTRAINDIC,    │
  │   ADVERTENCIA, etc.)                     │
  │ • Normaliza texto                       │
  │ • Divide em chunks de ≤512 tokens       │
  │ • Gera IDs únicos por chunk              │
  └─────────────────┬───────────────────────┘
                    ▼
          data/chunks_bulas.jsonl
          ~156.000 chunks
          {"id", "texto", "medicamento", "fonte"}

          ┌─────────────────────────────────┐
          │ scripts/embeddings.py           │
          │ + SentenceTransformer (MINILM) │
          │ • Gera vetores 384d por chunk   │
          │ • Indexa no ChromaDB            │
          └─────────────────┬───────────────┘
                            ▼
                  data/chroma_db/
                  ChromaDB collection
                  155.987 vetores × 384d

================================================================

ANOTAÇÕES — manuais.csv
  239 contextos × pares droga+droga+classe
  Classe: 0 (nenhuma), 1 (moderada), 2 (grave)

  ┌─────────────────────────────────────────┐
  │ GroupKFold por contexto                 │
  │ (cada contexto → 1 split só)           │
  └─────────────────┬───────────────────────┘
                    ▼
          train.csv  (850 pares, 190 ctx)
          val.csv    (105 pares,  24 ctx)
          test.csv   (105 pares,  24 ctx)

          ┌─────────────────────────────────┐
          │ scripts/train_classifier.py     │
          │ Fine-tuning pucpr/biobertpt-all │
          │ [CLS] alvo [SEP] outro [SEP] ctx │
          │ class_weights=[1, 2, 3]          │
          │ FP16, lr=2e-5, epochs=3         │
          └─────────────────┬───────────────┘
                            ▼
          data/modelos_finetuned/
          biobertpt-interactions/
            config.json
            model.safetensors (pesos finais)
            tokenizer/

================================================================

MODO USO — RAG Pipeline (scripts/rag.py)

  Query: "Paciente renal pode usar ibuprofeno com enalapril?"
          │
          ▼
  ┌───────────────────┐
  │ scripts/ner.py    │  MedicationNER
  │ clinicalnerpt-    │  Extrai: ibuprofeno, enalapril
  │ chemical          │
  └─────────┬─────────┘
            ▼
  ┌───────────────────────────┐
  │ scripts/embeddings.py      │  Busca vetorial
  │ ChromaDB                   │  Top-5 chunks mais similares
  │ Busca: ibuprofeno +       │
  │ enalapril + rim           │
  └─────────────┬─────────────┘
                ▼
  ┌───────────────────────────┐
  │ Monta contexto RAG         │
  │ "Trechos de bulas mais     │
  │  relevantes: ..."          │
  └─────────────┬─────────────┘
                ▼
  ┌───────────────────────────┐
  │ LLM (DeepSeek API ou       │
  │ GPT4All local)             │
  │ Prompt: papel + contexto  │
  │ + chunks + instrução      │
  │ → JSON {classe, justificativa}
  └───────────────────────────┘
```

---

## Classes de Interação

O modelo classifica em 3 classes:

| Classe | Nome | Significado | Exemplo |
|--------|------|-------------|---------|
| **0** | Nenhuma | Não há interação relevante documentada | Paracetamol + Amoxicilina |
| **1** | Moderada | Pode exigir monitoramento ou ajuste de dose | Sinvastatina + Amiodarona |
| **2** | Grave | Contraindicada ou pode causar dano sério | Warfarina + Aspirina em alta dose |
