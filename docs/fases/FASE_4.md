# Fase 4: Fine-Tuning do BioBERTpt

**Objetivo:** Fazer fine-tuning do `pucpr/biobertpt-all` para classificação de interações medicamentosas em 3 classes, usando o dataset anotado na Fase 3. Treinar em GPU, avaliar com F1-score por classe, e salvar o modelo para uso no pipeline RAG.

**Dependências:** Fase 3 concluída (dataset `train.csv` + `val.csv` + `test.csv`).

---

## Arquitetura do Fine-Tuning

```
pucpr/biobertpt-all (BERT base, 110M params)
        │
        ▼
[CLS] alvo [SEP] outro [SEP] contexto [SEP]
        │
        ▼
    Pooler Output (768 dims)
        │
        ▼
   Dropout (0.3)
        │
        ▼
   Linear(768 → 3) + Softmax
        │
        ▼
   Classe: 0 (SEM), 1 (LEVE), 2 (GRAVE)
```

### Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Batch size | 16 | Cabe folgado em 6 GB VRAM com FP16 |
| Gradient accumulation | 4 | Batch efetivo = 64 sem estourar VRAM |
| Learning rate | 2e-5 | Padrão para fine-tuning BERT |
| Épocas | 3 | Early stopping se val_loss não melhorar |
| Warmup steps | 10% do total | Evita oscilação inicial |
| Weight decay | 0.01 | Regularização |
| Dropout | 0.3 | Prevenir overfitting (dataset pequeno) |
| Max seq length | 256 | Suficiente para [CLS] + alvo + outro + contexto |
| FP16 | True | Metade da VRAM, ~2x mais rápido em RTX 3050 |
| Class weights | [1.0, 2.0, 3.0] | Penaliza mais erros em classes graves |

---

## Tarefas

### 4.1 Criar `scripts/train_classifier.py`

- [ ] `class InteractionDataset(Dataset)`
  - Construtor: carrega CSV, tokeniza com `AutoTokenizer`
  - Template: `[CLS] {medicamento_alvo} [SEP] {medicamento_outro} [SEP] {contexto} [SEP]`
  - Retorna: `input_ids`, `attention_mask`, `label`

- [ ] Função `carregar_dados(train_path, val_path, test_path) -> tuple[Dataset, Dataset, Dataset]`
  - Carrega CSVs da Fase 3
  - Aplica tokenização com `max_length=256`, `padding="max_length"`, `truncation=True`
  - Loga: tamanho de cada split, distribuição de classes

- [ ] Função `criar_modelo(num_classes=3) -> AutoModelForSequenceClassification`
  - `AutoModelForSequenceClassification.from_pretrained("pucpr/biobertpt-all", num_labels=3)`
  - Configurar `classifier_dropout = 0.3`
  - Mover para GPU

- [ ] Função `treinar(modelo, train_loader, val_loader, config) -> dict`
  - Training loop com `accelerate` ou manual
  - Metric: F1-score (macro + por classe), accuracy, loss
  - Early stopping: paciência = 2 épocas sem melhora no val_f1
  - Salvar melhor checkpoint em `data/modelos_finetuned/biobertpt-interactions/`
  - Logar curva de aprendizado (train_loss, val_loss, val_f1 por época)

- [ ] Função `avaliar(modelo, test_loader) -> dict`
  - Carregar melhor checkpoint
  - Métricas no teste: accuracy, precision/recall/F1 por classe, matriz de confusão
  - Análise de erros: top-20 erros (onde modelo errou com alta confiança)

### 4.2 Script principal

- [ ] `if __name__ == "__main__"`:
  1. Verificar GPU disponível
  2. Carregar dados
  3. Criar modelo
  4. Treinar (3 épocas, early stopping)
  5. Avaliar no teste
  6. Salvar métricas em `logs/training_metrics.json`
  7. Salvar matriz de confusão como `logs/confusion_matrix.png`

### 4.3 Criar `scripts/classifier.py` (wrapper de inferência)

- [ ] `class InteractionClassifier`
  - `__init__`: carrega modelo fine-tuned + tokenizer
  - `classificar(medicamento_alvo, medicamento_outro, contexto) -> dict`
    - Tokeniza entrada
    - Inferência em GPU (sem gradiente)
    - Retorna: `{"classe": int, "confianca": float, "probabilidades": [float, float, float]}`
  - `classificar_lote(pares: list[dict]) -> list[dict]`
    - Batch inference para eficiência

### 4.4 Testes

- [ ] `tests/test_train_classifier.py`:
  - [ ] `test_interaction_dataset` — cria dataset de amostra, verifica shapes
  - [ ] `test_tokenization_template` — `[CLS] alvo [SEP] outro [SEP] contexto [SEP]` presente
  - [ ] `test_modelo_criado` — `num_labels=3`, classifier head existe
  - [ ] `test_class_weights` — classe 2 tem peso ≥ 2x da classe 0

- [ ] `tests/test_classifier.py`:
  - [ ] `test_classificar_grave` — par grave → classe 2, confiança > 0.5
  - [ ] `test_classificar_sem_interacao` — par sem interação → classe 0
  - [ ] `test_classificar_lote` — 4 pares → 4 resultados
  - [ ] `test_output_format` — chaves `classe`, `confianca`, `probabilidades`

### 4.5 Métricas esperadas

- [ ] F1 macro ≥ 0.75
- [ ] F1 classe 2 (GRAVE) ≥ 0.70 (classe mais importante)
- [ ] F1 classe 0 (SEM) ≥ 0.80
- [ ] Matriz de confusão: poucos falsos negativos na classe 2

---

## Artefatos Produzidos

```
scripts/train_classifier.py
scripts/classifier.py
tests/test_train_classifier.py
tests/test_classifier.py
data/modelos_finetuned/biobertpt-interactions/   (modelo salvo)
logs/training_metrics.json
logs/confusion_matrix.png
```

---

## Verificação

- [ ] `python scripts/train_classifier.py` — treina sem OOM, salva checkpoint
- [ ] `python -m pytest tests/test_classifier.py -v` — todos passam
- [ ] F1 macro ≥ 0.75 no conjunto de teste
- [ ] Modelo carrega: `classifier = InteractionClassifier()` sem erros
- [ ] Inferência < 100ms por par em GPU
- [ ] Commit: `git add scripts/train_classifier.py scripts/classifier.py tests/test_*.py && git commit -m "feat: Fase 4 — fine-tuning BioBERTpt — F1 macro ≥ 0.75"`
