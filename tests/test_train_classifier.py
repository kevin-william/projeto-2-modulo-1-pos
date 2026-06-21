"""Testes para o script de treinamento do classificador."""

import tempfile
from pathlib import Path

import pandas as pd
import torch
import pytest
from transformers import AutoTokenizer

from scripts.config import CLASSES, CLASSIFIER_MODEL, DEVICE
from scripts.train_classifier import (
    InteractionDataset,
    carregar_dados,
    criar_modelo,
    _compute_metrics,
)

import numpy as np


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)


@pytest.fixture
def sample_data(tmp_path):
    """Cria splits de treino/val/teste em arquivos CSV temporarios."""
    data = []
    for i in range(30):
        c = i % 3
        data.append({
            "medicamento_alvo": f"alvo_{i}",
            "medicamento_outro": f"outro_{i}",
            "contexto": f"Contexto de interacao numero {i} entre medicamentos.",
            "classe": c,
            "fonte": "fonte1",
            "origem": "manual",
            "confianca": 1.0,
        })
    df = pd.DataFrame(data)
    train = df.iloc[:20]
    val = df.iloc[20:25]
    test = df.iloc[25:]

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    return train_path, val_path, test_path


class TestInteractionDataset:
    """Testes para o dataset tokenizado."""

    def test_len(self, tokenizer, sample_data):
        train_path, _, _ = sample_data
        train_df = pd.read_csv(train_path)
        ds = InteractionDataset(train_df, tokenizer)
        assert len(ds) == len(train_df)

    def test_shapes(self, tokenizer, sample_data):
        train_path, _, _ = sample_data
        train_df = pd.read_csv(train_path)
        ds = InteractionDataset(train_df, tokenizer)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item
        assert item["label"].item() in (0, 1, 2)

    def test_input_ids_not_empty(self, tokenizer, sample_data):
        train_path, _, _ = sample_data
        train_df = pd.read_csv(train_path)
        ds = InteractionDataset(train_df, tokenizer)
        item = ds[0]
        assert item["input_ids"].numel() > 0
        assert item["attention_mask"].sum() > 0  # pelo menos alguns tokens reais

    def test_labels_correspond(self, tokenizer, sample_data):
        train_path, _, _ = sample_data
        train_df = pd.read_csv(train_path)
        ds = InteractionDataset(train_df, tokenizer)
        # Verifica que cada label corresponde ao CSV original
        for i in range(min(5, len(ds))):
            item = ds[i]
            assert item["label"].item() == int(train_df.iloc[i]["classe"])


class TestCarregarDados:
    """Testes para a funcao carregar_dados."""

    def test_carregar(self, tokenizer, sample_data):
        train_path, val_path, test_path = sample_data
        train_ds, val_ds, test_ds, stats = carregar_dados(train_path, val_path, test_path, tokenizer)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert "train" in stats
        assert stats["train"]["n"] == 20


class TestCriarModelo:
    """Testes para a funcao criar_modelo."""

    def test_num_labels(self):
        modelo = criar_modelo(num_classes=3, dropout=0.3)
        assert modelo.config.num_labels == 3

    def test_classifier_head(self):
        modelo = criar_modelo(num_classes=3)
        assert hasattr(modelo, "classifier") or hasattr(modelo, "score")
        # Move para CPU para verificacao
        modelo.cpu()
        # O modelo tem uma cabeca de classificacao Linear(768, 3)
        n_params = sum(p.numel() for p in modelo.parameters())
        assert n_params > 1e7  # > 10M params = BERT base

    def test_dropout_config(self):
        modelo = criar_modelo(num_classes=3, dropout=0.3)
        assert modelo.config.classifier_dropout == 0.3


class TestComputeMetrics:
    """Testes para a funcao _compute_metrics."""

    def test_perfect(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 0, 1, 1, 2, 2])
        metrics = _compute_metrics(labels, preds)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_baseline(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        preds = np.array([0, 0, 1, 1, 2, 2])
        metrics = _compute_metrics(labels, preds)
        assert 0.0 < metrics["accuracy"] < 1.0
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 3
