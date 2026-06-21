"""
Treinamento e avaliacao do classificador de interacoes medicamentosas.

Faz fine-tuning do pucpr/biobertpt-all para 3 classes usando
os dados anotados da Fase 3 (train/val/test.csv).

Arquitetura:
    [CLS] alvo [SEP] outro [SEP] contexto [SEP]
    → BioBERTpt (110M) → Pooler (768 dims)
    → Dropout(0.3) → Linear(768→3) + Softmax

Hiperparametros:
    Batch=16, GradAccum=4, LR=2e-5, Epochs=3 (early stopping)
    FP16, ClassWeights=[1.0, 2.0, 3.0]

Uso:
    python scripts/train_classifier.py
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from scripts.config import (
    ANOTACOES_DIR,
    BATCH_SIZE,
    CLASS_WEIGHTS,
    CLASSIFIER_MODEL,
    CLASSES,
    DEVICE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGS_DIR,
    MAX_SEQ_LENGTH,
    MODELOS_DIR,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

log = logging.getLogger(__name__)


# ─── Dataset ────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """Dataset para pares medicamentosos tokenizados."""

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        alvo = str(row["medicamento_alvo"])
        outro = str(row["medicamento_outro"])
        contexto = str(row["contexto"])
        label = int(row["classe"])

        # Template: [CLS] alvo [SEP] outro [SEP] contexto [SEP]
        text = f"{alvo} [SEP] {outro} [SEP] {contexto}"

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ─── Funcoes auxiliares ─────────────────────────────────────────

def carregar_dados(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    tokenizer: AutoTokenizer,
) -> Tuple[InteractionDataset, InteractionDataset, InteractionDataset, dict]:
    """Carrega e tokeniza os splits de dados."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    stats = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = df["classe"].value_counts().sort_index().to_dict()
        stats[name] = {"n": len(df), "distribuicao": {CLASSES.get(k, str(k)): v for k, v in dist.items()}}
        log.info("  %s: %d pares | %s", name, len(df), {CLASSES.get(k, str(k)): v for k, v in dist.items()})

    train_ds = InteractionDataset(train_df, tokenizer)
    val_ds = InteractionDataset(val_df, tokenizer)
    test_ds = InteractionDataset(test_df, tokenizer)
    return train_ds, val_ds, test_ds, stats


def criar_modelo(num_classes: int = 3, dropout: float = DROPOUT) -> AutoModelForSequenceClassification:
    """Cria o modelo BioBERTpt com cabeca de classificacao."""
    id2label = {0: "SEM_INTERACAO", 1: "LEVE_MODERADA", 2: "GRAVE_CONTRAINDICADA"}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.classifier_dropout = dropout
    model.to(DEVICE)
    log.info("Modelo carregado: %s (%.0fM params) em %s", CLASSIFIER_MODEL,
             sum(p.numel() for p in model.parameters()) / 1e6, DEVICE)
    return model


def _compute_metrics(
    labels: np.ndarray, preds: np.ndarray,
) -> dict:
    """Calcula metricas de avaliacao."""
    accuracy = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_per_class": {str(c): round(f, 4) for c, f in zip([0, 1, 2], f1_per_class)},
        "precision_per_class": {str(c): round(p, 4) for c, p in zip([0, 1, 2], precision_per_class)},
        "recall_per_class": {str(c): round(r, 4) for c, r in zip([0, 1, 2], recall_per_class)},
        "support_per_class": {str(c): int(s) for c, s in zip([0, 1, 2], support)},
        "confusion_matrix": cm.tolist(),
    }


# ─── Treinamento ────────────────────────────────────────────────

def treinar(
    modelo: AutoModelForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
) -> dict:
    """Loop de treinamento com early stopping e gradient accumulation."""

    # Optimizer
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Class weights
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Scalers / tracking
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    use_amp = scaler is not None

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_accuracy": []}

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ─── Train ───
        modelo.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = modelo(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            else:
                outputs = modelo(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Handle last incomplete accumulation
        if step % GRADIENT_ACCUMULATION_STEPS != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(round(avg_train_loss, 4))

        # ─── Val ───
        val_loss, _, val_preds, val_labels = _evaluate_loader(modelo, val_loader, loss_fn)
        val_metrics = _compute_metrics(val_labels, val_preds)
        val_f1 = val_metrics["f1_macro"]

        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["val_accuracy"].append(round(val_metrics["accuracy"], 4))

        log.info("Epoca %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val F1: %.4f | Val Acc: %.4f",
                 epoch, EPOCHS, avg_train_loss, val_loss, val_f1, val_metrics["accuracy"])

        # ─── Checkpoint ───
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Salvar melhor modelo
            modelo.save_pretrained(output_dir)
            tokenizer_local = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
            tokenizer_local.save_pretrained(output_dir)
            log.info("  ✓ Melhor modelo salvo (F1=%.4f) em %s", val_f1, output_dir)
        else:
            patience_counter += 1
            log.info("  Sem melhora (patience %d/%d)", patience_counter, EARLY_STOPPING_PATIENCE)
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log.info("Early stopping na epoca %d", epoch)
                break

    history["best_epoch"] = best_epoch
    history["best_val_f1"] = round(best_val_f1, 4)
    return history


def _evaluate_loader(
    modelo: AutoModelForSequenceClassification,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Avalia modelo em um DataLoader."""
    modelo.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = modelo(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_labels)


def avaliar(modelo: AutoModelForSequenceClassification, test_loader: DataLoader, output_dir: Path) -> dict:
    """Avalia o modelo no conjunto de teste."""
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    test_loss, preds, labels, _ = _evaluate_loader(modelo, test_loader, loss_fn)
    metrics = _compute_metrics(labels, preds)
    metrics["test_loss"] = round(test_loss, 4)

    log.info("=" * 60)
    log.info("RESULTADO FINAL — CONJUNTO DE TESTE")
    log.info("=" * 60)
    log.info("Accuracy: %.4f", metrics["accuracy"])
    log.info("F1 Macro: %.4f", metrics["f1_macro"])
    log.info("Precision Macro: %.4f", metrics["precision_macro"])
    log.info("Recall Macro: %.4f", metrics["recall_macro"])
    log.info("F1 por classe: 0=%.4f  1=%.4f  2=%.4f",
             metrics["f1_per_class"]["0"], metrics["f1_per_class"]["1"], metrics["f1_per_class"]["2"])
    log.info("Matriz de confusao:\n%s",
             np.array2string(np.array(metrics["confusion_matrix"])))

    # Analise de erros: top-20 erros com maior confianca
    modelo.eval()
    erros = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["label"].to(DEVICE)
            outputs = modelo(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds_batch = probs.argmax(dim=-1)
            confs = probs.max(dim=-1).values

            mask_erro = preds_batch != labels_batch
            for i in range(len(labels_batch)):
                if mask_erro[i]:
                    erros.append({
                        "true": int(labels_batch[i]),
                        "pred": int(preds_batch[i]),
                        "confidence": round(confs[i].item(), 4),
                    })

    erros.sort(key=lambda x: x["confidence"], reverse=True)
    metrics["top_erros"] = erros[:20]

    if erros:
        log.info("\nTop-5 erros com maior confianca:")
        for i, e in enumerate(erros[:5], 1):
            log.info("  %d. true=%d pred=%d conf=%.4f", i, e["true"], e["pred"], e["confidence"])

    return metrics


# ─── Main ──────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log.info("=" * 60)
    log.info("Fine-Tuning BioBERTpt — Interacoes Medicamentosas")
    log.info("=" * 60)
    log.info("Dispositivo: %s", DEVICE)
    log.info("Batch size: %d (efetivo=%d)", BATCH_SIZE, BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    log.info("LR: %e | Epochs: %d | Dropout: %.1f", LEARNING_RATE, EPOCHS, DROPOUT)
    log.info("Class weights: %s", CLASS_WEIGHTS)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)

    # Carregar dados
    log.info("\nCarregando dados...")
    train_path = ANOTACOES_DIR / "train.csv"
    val_path = ANOTACOES_DIR / "val.csv"
    test_path = ANOTACOES_DIR / "test.csv"
    train_ds, val_ds, test_ds, stats = carregar_dados(train_path, val_path, test_path, tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Criar modelo
    log.info("\nCriando modelo...")
    modelo = criar_modelo()

    # Treinar
    log.info("\nIniciando treinamento...")
    output_dir = MODELOS_DIR / "biobertpt-interactions"
    history = treinar(modelo, train_loader, val_loader, output_dir)

    # Avaliar no teste (carregar melhor checkpoint)
    log.info("\nAvaliando no teste...")
    modelo.eval()
    modelo = AutoModelForSequenceClassification.from_pretrained(output_dir)
    modelo.to(DEVICE)
    metrics = avaliar(modelo, test_loader, output_dir)

    # Salvar metricas
    result = {
        "timestamp": datetime.now().isoformat(),
        "dispositivo": DEVICE,
        "hiperparametros": {
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "max_seq_length": MAX_SEQ_LENGTH,
            "class_weights": CLASS_WEIGHTS,
        },
        "dados": {
            "train": stats["train"]["n"],
            "val": stats["val"]["n"],
            "test": stats["test"]["n"],
        },
        "treinamento": history,
        "avaliacao": {k: v for k, v in metrics.items() if k != "top_erros"},
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = LOGS_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("\nMetricas salvas em: %s", metrics_path)

    # Verificar metas
    f1_macro = metrics["f1_macro"]
    f1_classe2 = metrics["f1_per_class"]["2"]
    f1_classe0 = metrics["f1_per_class"]["0"]
    log.info("\n=== VERIFICACAO DE METAS ===")
    log.info("F1 macro: %.4f (meta: >= 0.75) %s", f1_macro, "✅" if f1_macro >= 0.75 else "❌")
    log.info("F1 classe 2 (GRAVE): %.4f (meta: >= 0.70) %s", f1_classe2, "✅" if f1_classe2 >= 0.70 else "❌")
    log.info("F1 classe 0 (SEM): %.4f (meta: >= 0.80) %s", f1_classe0, "✅" if f1_classe0 >= 0.80 else "❌")
    log.info("\nConcluido.")


if __name__ == "__main__":
    main()
