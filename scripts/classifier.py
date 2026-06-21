"""
Wrapper de inferencia para o classificador BioBERTpt fine-tuned.

Carrega o modelo treinado e oferece interface para classificar
pares medicamentosos individualmente ou em lote.

Uso:
    from scripts.classifier import InteractionClassifier

    clf = InteractionClassifier()
    result = clf.classificar("amoxicilina", "ibuprofeno", "...contexto...")
    # {"classe": 1, "confianca": 0.87, "probabilidades": [0.05, 0.87, 0.08]}

    results = clf.classificar_lote(pares)
    # [{"classe": 1, ...}, {"classe": 0, ...}, ...]
"""

import os
from dotenv import load_dotenv
load_dotenv()  # carrega HF_TOKEN do .env
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.config import (
    CLASSES,
    CLASSIFIER_MODEL,
    DEVICE,
    MAX_SEQ_LENGTH,
    MODELOS_DIR,
)

log = logging.getLogger(__name__)


class InteractionClassifier:
    """Classificador de interacoes medicamentosas com BioBERTpt fine-tuned.

    Attributes:
        modelo: Modelo AutoModelForSequenceClassification carregado do checkpoint.
        tokenizer: Tokenizer do BioBERTpt.
        device: Dispositivo de inferencia (cuda/cpu).
        classes: Mapeamento id → nome da classe.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Caminho para o checkpoint fine-tuned.
                        Default: data/modelos_finetuned/biobertpt-interactions/
                        Se nao existir, carrega o modelo base sem fine-tuning.
            device: Dispositivo ('cuda', 'cpu', ou None para auto-deteccao).
        """
        self.device = device or DEVICE
        self.classes = CLASSES

        # Resolver caminho do modelo
        if model_path is None:
            finetuned = MODELOS_DIR / "biobertpt-interactions"
            if finetuned.exists() and (finetuned / "config.json").exists():
                model_path = str(finetuned)
                log.info("Checkpoint fine-tuned encontrado: %s", model_path)
            else:
                model_path = CLASSIFIER_MODEL
                log.info("Checkpoint nao encontrado, usando modelo base: %s", model_path)

        # Tokenizer: usar o modelo base, nao o checkpoint fine-tuned
        tokenizer_path = CLASSIFIER_MODEL if str(MODELOS_DIR) in model_path else model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except ValueError:
            self.tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
            log.info("Tokenizer fallback para modelo base: %s", CLASSIFIER_MODEL)

        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3,
            id2label={0: "SEM_INTERACAO", 1: "LEVE_MODERADA", 2: "GRAVE_CONTRAINDICADA"},
            label2id={"SEM_INTERACAO": 0, "LEVE_MODERADA": 1, "GRAVE_CONTRAINDICADA": 2},
            ignore_mismatched_sizes=True,
        )
        self.modelo.to(self.device)
        self.modelo.eval()

        n_params = sum(p.numel() for p in self.modelo.parameters())
        log.info("Classifier carregado (%s): %.0fM params em %s", model_path, n_params / 1e6, self.device)

    def classificar(
        self,
        medicamento_alvo: str,
        medicamento_outro: str,
        contexto: str,
    ) -> dict:
        """Classifica um par medicamentoso.

        Args:
            medicamento_alvo: Principio ativo da bula.
            medicamento_outro: Medicamento potencialmente interagente.
            contexto: Trecho da bula que descreve a interacao.

        Returns:
            Dict com chaves:
                - classe: int (0, 1 ou 2)
                - nome_classe: str ("SEM_INTERACAO", "LEVE_MODERADA", "GRAVE_CONTRAINDICADA")
                - confianca: float (0.0 a 1.0)
                - probabilidades: list[float] (prob. de cada classe)
        """
        # Template: [CLS] alvo [SEP] outro [SEP] contexto [SEP]
        texto = f"{medicamento_alvo} [SEP] {medicamento_outro} [SEP] {contexto}"

        encoded = self.tokenizer(
            texto,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.modelo(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            classe = int(outputs.logits.argmax(dim=-1).item())
            confianca = float(probs[0, classe].item())
            probabilidades = probs[0].tolist()

        return {
            "classe": classe,
            "nome_classe": self.classes.get(classe, f"DESCONHECIDA_{classe}"),
            "confianca": round(confianca, 4),
            "probabilidades": [round(p, 4) for p in probabilidades],
        }

    def classificar_lote(self, pares: List[dict]) -> List[dict]:
        """Classifica um lote de pares de forma eficiente.

        Args:
            pares: Lista de dicts com chaves:
                   medicamento_alvo, medicamento_outro, contexto.

        Returns:
            Lista de resultados, mesma ordem da entrada.
        """
        if not pares:
            return []

        textos = []
        for p in pares:
            alvo = p["medicamento_alvo"]
            outro = p["medicamento_outro"]
            ctx = p["contexto"]
            textos.append(f"{alvo} [SEP] {outro} [SEP] {ctx}")

        encoded = self.tokenizer(
            textos,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.modelo(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            classes = outputs.logits.argmax(dim=-1)
            confs = probs.gather(1, classes.unsqueeze(1)).squeeze(1)

        results = []
        for i in range(len(pares)):
            c = int(classes[i].item())
            results.append({
                "classe": c,
                "nome_classe": self.classes.get(c, f"DESCONHECIDA_{c}"),
                "confianca": round(float(confs[i].item()), 4),
                "probabilidades": [round(float(p), 4) for p in probs[i].tolist()],
            })

        return results
