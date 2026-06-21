"""
Configuração global do Detector de Interações Medicamentosas.

Centraliza paths, modelos, hiperparâmetros e constantes usadas por todos os módulos.
"""

import re
from pathlib import Path
import torch

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BULAS_F1_DIR = DATA_DIR / "bulas" / "fonte1"
BULAS_F2_DIR = DATA_DIR / "bulas" / "fonte2"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
ANOTACOES_DIR = DATA_DIR / "anotacoes"
MODELOS_DIR = DATA_DIR / "modelos_finetuned"
LOGS_DIR = PROJECT_ROOT / "logs"

# ─── Dispositivo ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Modelos Hugging Face ─────────────────────────────────────────────
NER_MODEL = "pucpr/clinicalnerpt-chemical"
CLASSIFIER_MODEL = "pucpr/biobertpt-all"
EMBEDDING_MODEL = "neuralmind/bert-base-portuguese-cased"
LLM_LOCAL_MODEL = "Phi-3-mini-4k-instruct.Q4_K_M.gguf"

# ─── Classes ──────────────────────────────────────────────────────────
CLASSES = {
    0: "SEM_INTERACAO",
    1: "LEVE_MODERADA",
    2: "GRAVE_CONTRAINDICADA",
}

# Pesos para compensar desbalanceamento (classe 2 = 3x mais penalidade)
CLASS_WEIGHTS = [1.0, 2.0, 3.0]

# ─── Parâmetros de Busca / RAG ────────────────────────────────────────
TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.75
BM25_ALPHA = 0.3  # peso cosseno vs BM25 na busca híbrida

# ─── Parâmetros de Fine-Tuning ────────────────────────────────────────
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
MAX_SEQ_LENGTH = 256
EARLY_STOPPING_PATIENCE = 2

# ─── Regex de Parseamento ─────────────────────────────────────────────
# Detecta se arquivo é Fonte 1 (começa com dígito) ou Fonte 2 (começa com letra)
FONTE1_PATTERN = re.compile(r"^\d+_")

# Seções da Fonte 1 a MANTER (fuzzy match com Levenshtein < 3)
SECOES_MANTER_F1 = {
    "contraindicacoes",
    "advertencias",
    "precaucoes",
    "advertencias e precaucoes",
    "interacoes medicamentosas",
    "interacoes  medicamentosas",
    "reacoes adversas",
    "efeitos adversos",
    "o que devo saber antes de usar",
    "quais os males que este medicamento pode causar",
}

# Blocos da Fonte 2 a MANTER (perguntas do Q&A)
BLOCOS_MANTER_F2 = [
    "INTERAÇÃO MEDICAMENTOSA?",
    "INTERAÇÃO MEDICAMENTOSA ?",
    "REAÇÕES ADVERSAS?",
    "PRECAUÇÕES?",
]

# ─── Palavras-chave para Weak Supervision ─────────────────────────────
GRAVE_KEYWORDS = [
    "contraindicado", "contra-indicado", "contraindicação",
    "fatal", "risco de morte", "nunca associar",
    "não administrar", "não deve ser administrado",
    "risco de vida", "arritmia fatal", "rabdomiólise",
    "hemorragia grave", "stevens-johnson",
    "insuficiência renal aguda", "interação severa",
    "toxicidade grave", "hepatotoxicidade",
]

LEVE_KEYWORDS = [
    "monitorar", "monitoramento", "monitorização",
    "ajustar dose", "ajuste de dose",
    "precaução", "cautela", "usar com cautela",
    "potencializa", "pode aumentar", "pode reduzir",
    "diminui efeito", "diminui a absorção",
    "recomenda-se", "deve ser monitorado",
    "pode interferir", "pode alterar",
    "recomenda-se acompanhamento",
]

SEM_INTERACAO_KEYWORDS = [
    "não há interação", "não há interações",
    "não foram observadas", "nenhuma interação",
    "não apresenta interação", "sem interação",
    "sem risco de interação", "é seguro",
    "pode ser usado", "pode ser administrado",
    "sem interação medicamentosa",
]

# ─── Estimativa de Tokens ─────────────────────────────────────────────
CHARS_PER_TOKEN = 4  # aproximação para português (1 token ≈ 4 caracteres)
