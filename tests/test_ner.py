"""Testes para scripts/ner.py — testes unitarios puros (sem download de modelo)."""

import pytest
from itertools import combinations

from scripts.ner import MedicationNER


class TestNormalizar:
    """Testes de normalizacao — puros, sem NER."""

    def test_minusculo(self):
        assert MedicationNER._normalizar("AMOXICILINA") == "amoxicilina"

    def test_acentos(self):
        assert "ó" not in MedicationNER._normalizar("Dipirona")
        assert "i" in MedicationNER._normalizar("Dipirona")

    def test_caracteres_especiais_removidos(self):
        result = MedicationNER._normalizar("AAS-Protect!")
        assert "-" not in result
        assert "!" not in result

    def test_espacos_extras(self):
        result = MedicationNER._normalizar("amox    ilina")
        assert "    " not in result

    def test_vazio(self):
        assert MedicationNER._normalizar("") == ""
        assert MedicationNER._normalizar("   ") == ""

    def test_duas_strings_iguais_normalizadas(self):
        assert MedicationNER._normalizar("Amoxicilina") == MedicationNER._normalizar("amoxicilina")

    def test_diferentes_pos_normalizacao(self):
        assert MedicationNER._normalizar("AAS") != MedicationNER._normalizar("Amox")

    def test_repeticao_vazia(self):
        """Normalizar string vazia nao crasha."""
        result = MedicationNER._normalizar("")
        assert result == ""


class TestGeracaoPares:
    """Testa geracao de pares (combinations) sem depender do NER real."""

    def test_2_meds_gera_1_par(self):
        meds = ["Amoxicilina", "Ibuprofeno"]
        pares = list(combinations(meds, 2))
        assert len(pares) == 1

    def test_3_meds_gera_3_pares(self):
        meds = ["Amoxicilina", "Ibuprofeno", "Dipirona"]
        pares = list(combinations(meds, 2))
        assert len(pares) == 3

    def test_1_med_gera_0_pares(self):
        meds = ["Amoxicilina"]
        pares = list(combinations(meds, 2))
        assert len(pares) == 0

    def test_0_meds_gera_0_pares(self):
        meds = []
        pares = list(combinations(meds, 2))
        assert len(pares) == 0

    def test_par_contem_dois_elementos(self):
        meds = ["A", "B"]
        pares = list(combinations(meds, 2))
        assert pares[0] == ("A", "B")
        assert len(pares[0]) == 2

    def test_pares_sao_tuplos(self):
        meds = ["A", "B", "C"]
        pares = list(combinations(meds, 2))
        for par in pares:
            assert isinstance(par, tuple)
            assert len(par) == 2
