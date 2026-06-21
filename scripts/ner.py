"""
Modulo de Reconhecimento de Entidades Medicamento (NER).

Usa o modelo clinicalnerpt-chemical para extrair medicamentos de texto
em linguagem natural. Suporta aggregation de sub-tokens (B-ChemicalDrugs
+ I-ChemicalDrugs → nome completo) e normalizacao para deduplicacao.

Fase 8 — Item: scripts/ner.py (RAGPipeline step 1).
"""

import os
from dotenv import load_dotenv
load_dotenv()  # carrega HF_TOKEN do .env
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from __future__ import annotations

import re
import logging
import unicodedata
from itertools import combinations
from typing import Optional

import pandas as pd

from scripts.config import DEVICE

log = logging.getLogger(__name__)

# Modelo NER — escolhe GPU se disponivel
NER_MODEL = "pucpr/clinicalnerpt-chemical"


class MedicationNER:
    """Reconhece medicamentos em texto usando clinicalnerpt-chemical.

    Attributes:
        model: pipeline transformers com modelo NER carregado
        device: "cuda" ou "cpu"
    """

    def __init__(self, model_name: str = NER_MODEL, device: Optional[str] = None):
        """Carrega o modelo NER.

        Args:
            model_name: nome do modelo no HuggingFace Hub
            device: "cuda", "cpu" ou None (auto-detecta)
        """
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Carregando NER model=%s device=%s", model_name, self.device)

        from transformers import pipeline
        self.model = pipeline(
            "ner",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple",
        )
        log.info("NER carregado com sucesso.")

    def extrair_medicamentos(self, texto: str) -> list[str]:
        """Extrai medicamentos de um texto.

        Agrega sub-tokens em nomes completos e normaliza para
        deduplicacao e comparacao.

        Args:
            texto: texto de entrada (ex: "Posso tomar Amoxicilina com Ibuprofeno?")

        Returns:
            Lista deduplicada de nomes de medicamentos encontrados
        """
        if not texto or not texto.strip():
            return []

        entidades = self.model(texto)
        medicamentos: list[str] = []

        i = 0
        while i < len(entidades):
            ent = entidades[i]

            # aggregation_strategy="simple" ja agrega B- e I- em um token
            # mas garantimos consistencia verificando entity_group
            grupo = ent.get("entity_group", ent.get("entity", ""))
            if grupo in ("ChemicalDrugs", "DRUG", "CHEMICAL", "Drug"):
                nome = ent["word"].strip()
                # Limpar prefixos de sub-word tokenizer
                nome = re.sub(r"^(##|Ġ|-LRB-|-RRB-)+", "", nome)
                nome = nome.strip()
                if nome and len(nome) > 1:
                    medicamentos.append(nome)
            i += 1

        # Normalizar e deduplicar
        normalizados = []
        seen_lower: set[str] = set()
        for med in medicamentos:
            norm = self._normalizar(med)
            if norm and norm not in seen_lower:
                seen_lower.add(norm)
                normalizados.append(med.strip())
            elif norm in seen_lower:
                # Ja visto — nao adicionar de novo
                pass

        log.debug("extrair_medicamentos: '%s' → %s", texto[:60], normalizados)
        return normalizados

    def extrair_pares(self, query: str) -> list[tuple[str, str]]:
        """Gera todos os pares de medicamentos de uma consulta.

        Args:
            query: texto da consulta do usuario

        Returns:
            Lista de tuplas (medicamento_alvo, medicamento_outro).
            Retorna lista vazia se < 2 medicamentos forem encontrados.
        """
        medicamentos = self.extrair_medicamentos(query)
        if len(medicamentos) < 2:
            log.debug("Menos de 2 medicamentos encontrados em: %s", query)
            return []

        pares = list(combinations(medicamentos, 2))
        log.debug("Pares gerados de '%s': %s", query[:40], pares)
        return pares

    @staticmethod
    def _normalizar(texto: str) -> str:
        """Normaliza nome do medicamento para comparacao.

        Lowercase + NFC + remove acentos e caracteres especiais.
        Mantem espacos internos (ex: "AAS Protect" → "aas protect").
        """
        if not texto:
            return ""
        texto = texto.lower().strip()
        texto = unicodedata.normalize("NFC", texto)
        # Remove acentos
        texto = "".join(
            c for c in texto
            if unicodedata.category(c) != "Mn"
        )
        texto = re.sub(r"[^a-z0-9\s]", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto


def testar_ner():
    """Teste rapido do NER."""
    ner = MedicationNER()

    consultas = [
        "Posso tomar Amoxicilina com Metotrexato?",
        "Dipirona e AAS juntos fazem mal?",
        "Paracetamol com Amoxicilina, pode?",
        "AAS Protect com Ibuprofeno é seguro?",
        "Amoxicilina, Ibuprofeno e Dipirona juntos?",
        "Posso beber álcool tomando Paracetamol?",
    ]

    print("\n" + "=" * 60)
    print("NER — Teste de Extracao de Medicamentos")
    print("=" * 60)

    for q in consultas:
        meds = ner.extrair_medicamentos(q)
        pares = ner.extrair_pares(q)
        print(f"\nQuery: {q}")
        print(f"  Medicamentos: {meds}")
        print(f"  Pares: {pares}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    testar_ner()
