"""Testes para o wrapper de inferencia do classificador."""

import pytest

from scripts.classifier import InteractionClassifier
from scripts.config import CLASSES


class TestInteractionClassifier:
    """Testes para o InteractionClassifier."""

    @pytest.fixture
    def classifier(self):
        """Retorna o classificador. Pode ser o modelo base se fine-tuned nao existir."""
        return InteractionClassifier()

    def test_classificar_output_format(self, classifier):
        result = classifier.classificar(
            medicamento_alvo="amoxicilina",
            medicamento_outro="ibuprofeno",
            contexto="Nao ha interacoes clinicamente relevantes com ibuprofeno.",
        )
        assert "classe" in result
        assert "nome_classe" in result
        assert "confianca" in result
        assert "probabilidades" in result
        assert result["classe"] in (0, 1, 2)
        assert result["nome_classe"] in list(CLASSES.values())
        assert 0.0 <= result["confianca"] <= 1.0
        assert len(result["probabilidades"]) == 3

    def test_classificar_lote(self, classifier):
        pares = [
            {
                "medicamento_alvo": "amoxicilina",
                "medicamento_outro": "ibuprofeno",
                "contexto": "Nao ha interacoes clinicamente relevantes com ibuprofeno.",
            },
            {
                "medicamento_alvo": "sinvastatina",
                "medicamento_outro": "itraconazol",
                "contexto": "O uso concomitante e contraindicado devido ao risco de rabdomiolise.",
            },
            {
                "medicamento_alvo": "losartana",
                "medicamento_outro": "hidroclorotiazida",
                "contexto": "Nao foram identificadas interacoes medicamentosas de importancia clinica.",
            },
            {
                "medicamento_alvo": "captopril",
                "medicamento_outro": "furosemida",
                "contexto": "Recomenda-se monitoramento da funcao renal.",
            },
        ]
        results = classifier.classificar_lote(pares)
        assert len(results) == 4
        for r in results:
            assert "classe" in r
            assert r["classe"] in (0, 1, 2)

    def test_contexto_vazio(self, classifier):
        result = classifier.classificar(
            medicamento_alvo="paracetamol",
            medicamento_outro="dipirona",
            contexto="",
        )
        assert result["classe"] in (0, 1, 2)
