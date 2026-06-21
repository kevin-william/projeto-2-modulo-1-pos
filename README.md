# Detector de Interações Medicamentosas com LLMs, NER e RAG

**Disciplina:** Sistemas Cognitivos com Large Language Models
**Aluno:** Kevin Rodrigues
**Professor:** _(preencher)_

---

## Descrição do Problema

Profissionais de saúde (médicos, pharmacists, enfermeiros) enfrentam diariamente o desafio de verificar interações medicamentosas antes de prescrever ou administrar fármacos. As bulas de medicamentos são documentos extensos — frequentemente com mais de 5.000 palavras e 10.000 tokens — e a seção de "Interações Medicamentosas" fica diluída entre dezenas de páginas de texto técnico. A verificação manual é lenta, propensa a erros e impraticável em ambientes de urgência.

Este projeto implementa um **sistema cognitivo end-to-end** que recebe uma consulta em linguagem natural (ex.: *"Posso tomar Amoxicilina com Metotrexato?"*), extrai os medicamentos mencionados usando NER, recupera trechos relevantes do bulário indexado em ChromaDB, classifica o par em três níveis de severidade (Sem Interação, Leve/Moderada, Grave/Contraindicada) usando um modelo BioBERTpt fine-tuned, e opcionalmente gera uma resposta fundamentada usando um LLM (GPT-4o-mini da OpenAI ou Phi-3-mini via GPT4All local).

---

## Arquitetura do Pipeline

```
Consulta do usuário (linguagem natural)
         │
         ▼
┌─────────────────────────────────────────┐
│  STAGE 1 — NER (pucpr/clinicalnerpt)  │
│  Extrai medicamentos do texto          │
│  Agrega tokens B-/I- em nomes completos │
│  Gera todos os pares (combinação 2 a 2)│
└────────────────┬──────────────────────┘
                 │ [lista de pares: (alvo, outro)]
                 ▼
┌─────────────────────────────────────────┐
│  STAGE 2 — Busca Vetorial (ChromaDB)   │
│  270.608 chunks de bulas indexados     │
│  Busca híbrida: 30% cosseno + 70% BM25 │
│  Top-5 chunks por par                  │
└────────────────┬──────────────────────┘
                 │ [par + chunks recuperados]
                 ▼
┌─────────────────────────────────────────┐
│  STAGE 3 — Classificador BioBERTpt     │
│  Fine-tuned em 374 pares anotados      │
│  3 classes: 0=SEM, 1=LEVE, 2=GRAVE    │
│  Atinge F1=0.73 na classe grave (meta)  │
└────────────────┬──────────────────────┘
                 │ [classe + confiança]
                 ▼
┌─────────────────────────────────────────┐
│  STAGE 4 — LLM (opcional)              │
│  OpenAI GPT-4o-mini (remoto)           │
│  ou GPT4All Phi-3-mini (local)        │
│  Gera resposta em português fundamentada│
└────────────────┬──────────────────────┘
                 ▼
        JSON estruturado
        {classe, confiança, evidência, fonte}
```

---

## Status das Fases

| Fase | Descrição | Status |
|------|-----------|--------|
| 0 | Setup + Configuração GPU + Dependências | ✅ Concluída |
| 1 | Pré-processamento de bulas (chunking, pruning) | ✅ Concluída |
| 2 | Notebook 01 — Modelos HF e NLP (5 tarefas) | ✅ Concluída |
| 3 | Anotação manual + revisão de 455 candidatos | ✅ Concluída |
| 4 | Fine-tuning BioBERTpt para classificação | ✅ Concluída |
| 5 | Notebook 02 — Prompt Engineering (5 técnicas) | ✅ Concluída |
| 6 | Notebook 03 — Embeddings + ChromaDB (270k chunks) | ✅ Concluída |
| 7 | Notebook 04 — Inferência Local vs Remota | ✅ Concluída |
| 8 | Notebook 05 — Pipeline RAG End-to-End | ✅ Concluída |
| 9 | Relatório PDF + README | ✅ Concluída |

---

## Estrutura do Projeto

```
projeto-2-modulo-1-pos/
├── c01_modelos_llm.ipynb          # Fase 2: 5 tarefas NLP + modelos HF
├── c02_prompting.ipynb           # Fase 5: prompt engineering
├── c03_embeddings_busca.ipynb   # Fase 6: embeddings + ChromaDB
├── c04_inferencia_local_ou_remota.ipynb  # Fase 7: GPT4All vs OpenAI
├── c05_rag_pipeline.ipynb        # Fase 8: pipeline RAG completo
│
├── scripts/
│   ├── config.py                 # Hyperparâmetros centralizados
│   ├── preprocess.py             # Chunking de bulas → JSONL
│   ├── embeddings.py             # Geração de embeddings + indexação ChromaDB
│   ├── annotate.py               # Interface de anotação manual
│   ├── validate_annotations.py    # Validação de anotações
│   ├── train_classifier.py       # Fine-tuning BioBERTpt
│   ├── classifier.py             # Wrapper de inferência do classificador
│   ├── ner.py                    # MedicationNER (pucpr/clinicalnerpt-chemical)
│   └── rag.py                    # RAGPipeline completo
│
├── tests/
│   ├── test_train_classifier.py  # Fine-tuning (13 testes)
│   ├── test_classifier.py        # Wrapper classificador (2 testes)
│   ├── test_ner.py               # NER — normalização e geracao de pares (13 testes)
│   └── test_rag.py              # RAG — sanitização, parsing, serialização (15 testes)
│
├── data/
│   ├── bulas/                    # Bulário bruto (não versionado)
│   │   ├── fonte1/              # 4.978 bulas ANVISA (.txt)
│   │   └── fonte2/              # 982 bulas Consultaremedios (.txt)
│   ├── chunks_bulas.jsonl        # 270.608 chunks indexados
│   ├── chroma_db/                # Índice vetorial ChromaDB (não versionado)
│   ├── modelos_finetuned/        # BioBERTpt fine-tuned (não versionado)
│   └── anotacoes/
│       ├── train.csv             # 374 pares de treino
│       ├── val.csv               # 94 pares de validação
│       └── test.csv              # 74 pares de teste
│
├── docs/
│   ├── PLANO_IMPLEMENTACAO.md    # Plano completo do projeto
│   ├── PASSO_A_PASSO_ANOTACAO.md # Guia de anotacao manual
│   ├── fases/
│   │   ├── FASE_3.md ... FASE_9.md
│   │   └── RUBRICAS.md
│   └── dataset/
│       └── RESUMO_BULAS.md       # Descricao do corpus
│
├── requirements.txt              # Dependências Python
├── .env.example                  # Template de variáveis de ambiente
├── .gitignore
└── README.md                     # Este arquivo
```

---

## Requisitos

- **Python:** 3.9 ou superior
- **RAM:** 8 GB (mínimo), 16 GB (recomendado)
- **GPU:** NVIDIA com 6 GB VRAM (opcional — funciona em CPU, mas mais lento)
- **Disco:** ~3 GB para modelos + ~500 MB para o bulário
- **Sistema:** Windows 10+ com Git Bash/MSYS, ou Linux/macOS

---

## Instalação

### 1. Clonar o repositório

```bash
git clone <url-do-repo>
cd projeto-2-modulo-1-pos
```

### 2. Criar ambiente virtual

```bash
# Windows (Git Bash / MSYS)
python -m venv venv
source venv/Scripts/activate

# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (CMD)
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. PyTorch com CUDA (recomendado para GPU NVIDIA)

```bash
# NVIDIA GPU com CUDA 12.4
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# NVIDIA GPU com CUDA 11.8
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU apenas (sem GPU)
pip install torch==2.6.0 torchvision torchaudio
```

### 5. Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha:

| Variável | Obrigatório | Onde obter | Custo |
|---|---|---|---|
| `HF_TOKEN` | **Recomendado** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Gratuito |
| `OPENAI_API_KEY` | Opcional | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Pay-per-use |

- **`HF_TOKEN`**: autentica downloads do HuggingFace Hub e elimina o aviso de rate-limiting. Nível mínimo: "Read". **Sem esta chave** o sistema funciona mas pode sofrer throttling.
- **`OPENAI_API_KEY`**: habilita GPT-4o-mini nos Notebooks 02, 04 e 05. **Sem esta chave** o sistema usa apenas o classificador fine-tuned e o GPT4All local (zero custo).

---

## Como Rodar Cada Etapa

### Fase 0 — Verificar Instalação

```bash
# Verifica GPU e CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU apenas')"
```

### Fase 1 — Pré-processamento de Bulas

```bash
# Coloca bulas brutas em data/bulas/fonte1/ e data/bulas/fonte2/
# Gera data/chunks_bulas.jsonl com 270.608 chunks
python -m scripts.preprocess
```

**Saída:** `data/chunks_bulas.jsonl` (~270.608 linhas, uma por chunk)

### Fase 2 — Notebook 01: Modelos HF e NLP

```bash
jupyter notebook c01_modelos_llm.ipynb
```

Demonstra 5 tarefas NLP: NER, classificação, QA, summarization, text generation, fill-mask usando modelos do Hugging Face. Não requer GPU.

### Fase 3 — Anotação Manual do Dataset

```bash
# Interface interativa de anotacao
python -m scripts.annotate

# Validacao das anotacoes
python -m scripts.validate_annotations
```

**Resultado:** `data/anotacoes/train.csv`, `val.csv`, `test.csv` (~374 treino, 94 validação, 74 teste)

### Fase 4 — Fine-tuning do Classificador

```bash
# Treina BioBERTpt fine-tuned em 3 epocas (GPU recomendada)
python -m scripts.train_classifier

# Avalia no conjunto de teste
python -m scripts.train_classifier --eval-only
```

**Resultado:** `data/modelos_finetuned/biobertpt-interactions/`
**Métricas obtidas:** F1 Classe 2 (Grave) = 0.73 ✅ | F1 Macro = 0.47 ⚠️

**Testes:**
```bash
python -m pytest tests/test_train_classifier.py tests/test_classifier.py -v
```

### Fase 5 — Notebook 02: Prompt Engineering

```bash
jupyter notebook c02_prompting.ipynb
```

Demonstra 3 técnicas de prompting: zero-shot, few-shot (3 exemplos), chain-of-thought. **Requer `OPENAI_API_KEY`** no `.env` para executar as células de avaliação. Células com a OpenAI mostram aviso e pulam se a chave não estiver configurada.

### Fase 6 — Notebook 03: Embeddings e ChromaDB

```bash
jupyter notebook c03_embeddings_busca.ipynb
```

Gera embeddings com 3 modelos (BERTpt, E5, MiniLM), indexa 270.608 chunks no ChromaDB, compara busca semântica vs híbrida. Executa em CPU (~10-15 min) ou GPU (~2-3 min).

**Script alternativo (linha de comando):**
```bash
# Gera embeddings e indexa no ChromaDB
python -m scripts.embeddings --model all-MiniLM-L6-v2 --chunks data/chunks_bulas.jsonl --output data/chroma_db
```

### Fase 7 — Notebook 04: Inferência Local vs Remota

```bash
jupyter notebook c04_inferencia_local_ou_remota.ipynb
```

Compara GPT4All (Phi-3-mini, local, gratuito, privado) vs OpenAI (GPT-4o-mini, remoto, ~R$ 0.0008/consulta). **Requer `OPENAI_API_KEY`** para o modo remoto. GPT4All local não requer chave.

### Fase 8 — Notebook 05: Pipeline RAG Completo

```bash
jupyter notebook c05_rag_pipeline.ipynb
```

Integra NER + ChromaDB + classificador fine-tuned + LLM em um pipeline end-to-end. Demonstra 8 consultas com JSON de saída, análise de chunking, segurança e falhas.

**Testes:**
```bash
python -m pytest tests/test_ner.py tests/test_rag.py -v
```

### Todos os Testes

```bash
python -m pytest tests/ -v
```

---

## Executar Sem GPU (CPU Apenas)

Todas as etapas funcionam em CPU. A única diferença é o tempo:

| Etapa | CPU | GPU (6 GB) |
|-------|-----|-------------|
| Pré-processamento | ~5 min | ~2 min |
| Fine-tuning | ~45 min | ~8 min |
| Geração de embeddings | ~15 min | ~3 min |
| NER por consulta | ~500 ms | ~80 ms |
| Classificação | ~100 ms | ~30 ms |

---

## Configuração de API (Opcional)

### HuggingFace (HF_TOKEN) — Recomendado

```bash
# Gere um token em: https://huggingface.co/settings/tokens
# Nível mínimo: "Read"
echo "HF_TOKEN=hf_seu_token" >> .env
```

### OpenAI (GPT-4o-mini)

```bash
# Editando .env manualmente
OPENAI_API_KEY=sk-proj-...   # sua chave da OpenAI
```

### GPT4All (local, gratuito)

Não requer chave. O modelo `Phi-3-mini-4k-instruct.gguf` (~2 GB) é baixado automaticamente na primeira execução.

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| `CUDA not available` | Reinstale PyTorch com CUDA: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` |
| `Out of memory` | Reduza `BATCH_SIZE` em `scripts/config.py` (padrão: 16 → 8) |
| `HF_TOKEN not set` / rate-limiting | Adicione `HF_TOKEN=hf_xxx` no `.env` — baixe em huggingface.co/settings/tokens |
| `dotenv not found` | `pip install python-dotenv` |
| `OPENAI_API_KEY not set` | Células OpenAI pulam com aviso — classificador fine-tuned e GPT4All local continuam funcionando |
| `Modelo não encontrado` | Verifique conexão com internet; para offline: `huggingface-cli download` |
| `ConnectionError` no GPT4All | Verifique rede ou use `fetch=False` se o modelo já estiver baixado |
| Testes falham no NER (mock) | Os testes do NER usam mocks e não precisam de download de modelo |

---

## Limitações

- **NER:** modelo `pucpr/clinicalnerpt-chemical` foi treinado em corpus clínico geral, não especificamente em bulas brasileiras. Nomes comerciais pouco comuns podem não ser reconhecidos.
- **Classificador:** fine-tuning feito com apenas ~374 pares anotados — limite de generalização para casos raros.
- **Cobertura:** limitada aos medicamentos presentes nas 5.960 bulas processadas.
- **Interações:** não cobre interações medicamento-alimento, medicamento-exame ou combinações com 3+ medicamentos simultâneos.
- **F1 Macro (0.47):** abaixo da meta de 0.75 — classe 0 (SEM interacao) colapsou durante o treino. Classe 2 (Grave) — a mais crítica para segurança — atingiu a meta (F1=0.73).

---

## Melhorias Futuras

- Fine-tuning do NER com anotações específicas de bulas brasileiras
- Expansão do dataset de anotação para ~2.000 pares
- Interface web com Streamlit ou Gradio
- Atualização automática do bulário com novas bulas ANVISA
- Suporte a interações medicamento-alimento e exame
- Avaliação com dados clínicos reais (após aprovação ética)

---

## Citação

Se usar este projeto, cite:

```
Rodrrigues, K. (2026). Detector de Interações Medicamentosas com LLMs, NER e RAG.
Disciplina de Sistemas Cognitivos com Large Language Models.
```

---

## Licença

Apenas para fins acadêmicos.
