# Fase 2: Notebook 01 — Modelos LLM e NLP (Hugging Face)

**Objetivo:** Demonstrar domínio do ecossistema Hugging Face com tarefas NLP aplicadas às bulas, seguindo o estilo do professor (pipeline, AutoTokenizer, AutoModel).

**Dependências:** Fase 0 (ambiente GPU) e Fase 1 (chunks disponíveis para teste).

**Rubricas cobertas:** Rubrica 1 — todos os 5 itens.

---

## Estrutura do Notebook

```
c01_modelos_llm.ipynb
├── 2.1  Setup e imports
├── 2.2  AutoModel + AutoTokenizer (estilo professor)
├── 2.3  Pipeline: sentiment-analysis em frases clínicas
├── 2.4  Pipeline: NER com clinicalnerpt-chemical
├── 2.5  Pipeline: text-generation com GPT-2 português
├── 2.6  Pipeline: fill-mask com BERT português
├── 2.7  Pipeline: summarization com BART
├── 2.8  Pipeline: question-answering com BERT pt
├── 2.9  Tabela comparativa de modelos
├── 2.10 Conclusão: quais tarefas importam para o detector
```

---

## Tarefas

### 2.1 Células de setup e imports

- [x] `import torch`, `from transformers import pipeline, AutoModel, AutoTokenizer`
- [x] `from scripts.config import *`
- [x] Verificar GPU: `assert torch.cuda.is_available()`
- [x] Exibir: `Device: {DEVICE}`, `GPU: {torch.cuda.get_device_name(0)}`

### 2.2 Demonstração: AutoModel + AutoTokenizer

- [x] Seguir EXATAMENTE o estilo do professor:
  ```python
  model_id = "pucpr/clinicalnerpt-chemical"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModel.from_pretrained(model_id)
  inputs = tokenizer("O mecanismo de atenção é poderoso", return_tensors="pt")
  outputs = model(**inputs)
  print(f"Dimensões do output: {outputs.last_hidden_state.shape}")
  ```
- [x] Célula Markdown: explicar dimensões `[Batch, Tokens, Hidden_Dim]`, o que são hidden states, tokenização WordPiece

### 2.3 Pipeline: sentiment-analysis

- [x] Usar `pipeline("sentiment-analysis")` (modelo padrão)
- [x] Testar com frases reais das bulas:
  - `"O uso concomitante é contraindicado"` → NEGATIVE
  - `"Não há interações conhecidas com este medicamento"` → POSITIVE
  - `"Recomenda-se monitoramento da função renal"` → analisar resultado
- [x] Célula Markdown: limitações — modelo genérico não entende nuance clínica; motivação para fine-tuning

### 2.4 Pipeline: NER com clinicalnerpt-chemical

- [x] `ner = pipeline("ner", model="pucpr/clinicalnerpt-chemical", aggregation_strategy="simple", device=0)`
- [x] Testar com trecho real de bula:
  ```
  "A administração concomitante de Amoxicilina com Metotrexato pode aumentar
   a toxicidade. Recomenda-se evitar o uso com Varfarina."
  ```
- [x] Exibir entidades: `Amoxicilina`, `Metotrexato`, `Varfarina` → todas `ChemicalDrugs`
- [x] Célula Markdown: NER como token classification, por que BERT (encoder-only), agregação de sub-tokens

### 2.5 Pipeline: text-generation com GPT-2 português

- [x] `gerador = pipeline("text-generation", model="pierreguillou/gpt2-small-portuguese")`
- [x] Prompt: `"Interação entre Amoxicilina e Ibuprofeno:"`
- [x] Mostrar saída (provável alucinação)
- [x] Variar parâmetros: `temperature=0.7`, `top_k=50`, `max_length=100`
- [x] Célula Markdown: decoder-only (GPT-2) vs encoder-only (BERT), alucinação, parâmetros de geração

### 2.6 Pipeline: fill-mask

- [x] `unmasker = pipeline("fill-mask", model="neuralmind/bert-base-portuguese-cased")`
- [x] Testar: `"O uso concomitante é [MASK] em casos de insuficiência renal"`
- [x] `top_k=5` — analisar tokens previstos
- [x] Célula Markdown: fill-mask revela conhecimento latente do BERT sobre relações entre palavras

### 2.7 Pipeline: summarization

- [x] `summarizer = pipeline("summarization", model="facebook/bart-large-cnn")`
- [x] Testar com seção de interações de uma bula real (Fonte 2, texto ~500 palavras)
- [x] `max_length=80, min_length=30, do_sample=False`
- [x] Célula Markdown: summarization abstrativa vs extrativa, desafios em português, BART como encoder-decoder

### 2.8 Pipeline: question-answering

- [x] `qa = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")`
- [x] Contexto: parágrafo da Fonte 1 sobre interações com Metotrexato
- [x] Pergunta: `"Quais medicamentos interagem com Amoxicilina?"`
- [x] Pergunta: `"Qual o risco da interação com Metotrexato?"`
- [x] Célula Markdown: QA extrativo — só funciona se resposta está literalmente no texto; limitação para perguntas inferenciais

### 2.9 Tabela comparativa de modelos

- [x] Tabela Markdown:

| Modelo | Arquitetura | Parâmetros | Tarefa | Limite Tokens | Domínio |
|---|---|---|---|---|---|
| `clinicalnerpt-chemical` | BERT (encoder-only) | 110M | NER | 512 | Clínico PT |
| `biobertpt-all` | BERT (encoder-only) | 110M | Classificação | 512 | Biomédico PT |
| `bart-large-cnn` | BART (encoder-decoder) | 406M | Summarization | 1024 | Genérico EN |
| `gpt2-small-portuguese` | GPT-2 (decoder-only) | 124M | Text Generation | 1024 | Genérico PT |
| `bert-base-portuguese-cased` | BERT (encoder-only) | 110M | Fill-mask / Embeddings | 512 | Genérico PT |

- [x] Célula Markdown: encoder-only (compreensão, classificação, NER) vs decoder-only (geração) vs encoder-decoder (tradução, sumarização). Pipeline vs inferência manual. Limite de 512 tokens do BERT e como contornar (chunking).

### 2.10 Conclusão

- [x] Resumo: quais tarefas são úteis para o detector de interações
  - NER → extrair medicamentos da consulta e das bulas
  - Classificação → classificar interações (com fine-tuning)
  - QA → extrair evidências textuais
  - Summarization → condensar seções longas (bônus)
- [x] Célula Markdown: próximos passos — anotação do dataset para fine-tuning

---

## Artefatos Produzidos

```
c01_modelos_llm.ipynb
```

---

## Verificação

- [x] Notebook executa do início ao fim sem erros (Kernel → Restart & Run All)
- [x] `torch.cuda.is_available()` → `True` em todas as células que usam GPU
- [x] Pipeline NER extrai pelo menos 2 entidades do trecho de teste
- [x] Tabela comparativa preenchida e explicada nas células Markdown
- [x] Células Markdown explicam cada pipeline no contexto do projeto
- [x] Commit: `git add c01_modelos_llm.ipynb && git commit -m "feat: Fase 2 — Notebook 01: modelos NLP com Hugging Face"`
