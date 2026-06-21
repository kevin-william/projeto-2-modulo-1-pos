# Fase 0: Estrutura de Diretórios e Setup GPU

**Objetivo:** Criar a estrutura base do projeto, instalar dependências, e configurar PyTorch com suporte CUDA para GPU.

**Dependências:** Nenhuma.

---

## Tarefas

### 0.1 Criar estrutura de diretórios

- [x] Criar `scripts/`, `tests/`
- [x] Criar `docs/fases/`, `docs/dataset/`
- [x] Criar `data/bulas/fonte1/`, `data/bulas/fonte2/`
- [x] Criar `data/chroma_db/`, `data/anotacoes/`, `data/modelos_finetuned/biobertpt-interactions/`
- [x] Criar `logs/`

### 0.2 Criar `scripts/__init__.py` e `tests/__init__.py`

- [x] Arquivos vazios para tornar os diretórios em pacotes Python

### 0.3 Criar ambiente virtual (venv)

- [x] `python -m venv venv`
- [x] Ativar: `source venv/Scripts/activate` (Git Bash) ou `venv\Scripts\activate` (CMD)
- [x] Verificar: `which python` → aponta para `.../projeto-2-modulo-1-pos/venv/Scripts/python`

### 0.4 Instalar PyTorch com CUDA no venv

- [x] `pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- [x] Verificar: `python -c "import torch; assert torch.cuda.is_available(), 'CUDA não disponível'; print(torch.cuda.get_device_name(0))"`
- [x] Resultado: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | PyTorch `2.5.1+cu124` | CUDA `12.4`

### 0.5 Criar `scripts/config.py`

Definir constantes globais:

- [x] `PROJECT_ROOT = Path(__file__).resolve().parent.parent`
- [x] `DATA_DIR`, `BULAS_F1_DIR`, `BULAS_F2_DIR`
- [x] `CHROMA_DB_DIR`, `ANOTACOES_DIR`, `MODELOS_DIR`
- [x] `NER_MODEL = "pucpr/clinicalnerpt-chemical"`
- [x] `CLASSIFIER_MODEL = "pucpr/biobertpt-all"`
- [x] `EMBEDDING_MODEL = "neuralmind/bert-base-portuguese-cased"`
- [x] `LLM_LOCAL_MODEL = "Phi-3-mini-4k-instruct.Q4_K_M.gguf"`
- [x] `TOP_K_CHUNKS = 5`, `SIMILARITY_THRESHOLD = 0.6`, `CONFIDENCE_THRESHOLD = 0.75`
- [x] `CLASSES = {0: "SEM_INTERACAO", 1: "LEVE_MODERADA", 2: "GRAVE_CONTRAINDICADA"}`
- [x] `CLASS_WEIGHTS = [1.0, 2.0, 3.0]` (pesos para compensar desbalanceamento)
- [x] `BATCH_SIZE = 16`, `LEARNING_RATE = 2e-5`, `EPOCHS = 3`
- [x] Regex para parseamento: `FONTE1_REGEX`, `SECOES_MANTER_F1`, `BLOCOS_MANTER_F2`
- [x] `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`

### 0.6 Criar `requirements.txt`

- [x] `torch==2.5.1`
- [x] `transformers>=4.40.0,<5.0`
- [x] `sentence-transformers>=2.7.0`
- [x] `chromadb>=0.5.0`
- [x] `gpt4all>=2.8.0`
- [x] `openai>=1.30.0`
- [x] `pandas>=2.0.0`
- [x] `rank-bm25>=0.2.2`
- [x] `python-dotenv>=1.0.0`
- [x] `pytest>=7.0`
- [x] `jupyter>=1.0.0`
- [x] `scikit-learn>=1.3.0`
- [x] `sentencepiece`, `sacremoses`
- [x] `accelerate>=0.30.0`

### 0.7 Criar `.env.example`

- [x] `OPENAI_API_KEY=***`
- [x] Comentário: "Copie para .env e preencha com sua chave real"

### 0.8 Criar `.gitignore`

- [x] `.env`
- [x] `venv/`
- [x] `data/bulas/`
- [x] `data/chroma_db/`
- [x] `data/modelos_finetuned/`
- [x] `logs/`
- [x] `__pycache__/`
- [x] `.ipynb_checkpoints/`
- [x] `*.gguf`

### 0.9 Verificar ambiente GPU

- [x] Executar script de diagnóstico: GPU, VRAM, CUDA, PyTorch
- [x] Registrar resultado em `logs/diagnostico_gpu.txt`
- [x] Resultado: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | VRAM `6.4 GB` | CUDA `12.4` | PyTorch `2.5.1+cu124`

---

## Artefatos Produzidos

```
scripts/__init__.py
scripts/config.py
tests/__init__.py
requirements.txt
.env.example
.gitignore
logs/diagnostico_gpu.txt
```

---

## Verificação

- [ ] `python -c "from scripts.config import DEVICE; print(f'Device: {DEVICE}')"` → `Device: cuda`
- [ ] `python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"` → `VRAM: 6.0 GB`
- [ ] `pip list | grep torch` mostra `torch 2.5.1` (não `+cpu`)
