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

- [ ] `import torch`, `from transformers import pipeline, AutoModel, AutoTokenizer`
- [ ] `from scripts.config import *`
- [ ] Verificar GPU: `assert torch.cuda.is_available()`
- [ ] Exibir: `Device: {DEVICE}`, `GPU: {torch.cuda.get_device_name(0)}`

### 2.2 Demonstração: AutoModel + AutoTokenizer

- [ ] Seguir EXATAMENTE o estilo do professor:
  ```python
  model_id = "pucpr/clinicalnerpt-chemical"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModel.from_pretrained(model_id)
  inputs = tokenizer("O mecanismo de atenção é poderoso", return_tensors="pt")
  outputs = model(**inputs)
  print(f"Dimensões do output: {outputs.last_hidden_state.shape}")
  ```
- [ ] Célula Markdown: explicar dimensões `[Batch, Tokens, Hidden_Dim]`, o que são hidden states, tokenização WordPiece

### 2.3 Pipeline: sentiment-analysis

- [ ] Usar `pipeline("sentiment-analysis")` (modelo padrão)
- [ ] Testar com frases reais das bulas:
  - `"O uso concomitante é contraindicado"` → NEGATIVE
  - `"Não há interações conhecidas com este medicamento"` → POSITIVE
  - `"Recomenda-se monitoramento da função renal"` → analisar resultado
- [ ] Célula Markdown: limitações — modelo genérico não entende nuance clínica; motivação para fine-tuning

### 2.4 Pipeline: NER com clinicalnerpt-chemical

- [ ] `ner = pipeline("ner", model="pucpr/clinicalnerpt-chemical", aggregation_strategy="simple", device=0)`
- [ ] Testar com trecho real de bula:
  ```
  "A administração concomitante de Amoxicilina com Metotrexato pode aumentar
   a toxicidade. Recomenda-se evitar o uso com Varfarina."
  ```
- [ ] Exibir entidades: `Amoxicilina`, `Metotrexato`, `Varfarina` → todas `ChemicalDrugs`
- [ ] Célula Markdown: NER como token classification, por que BERT (encoder-only), agregação de sub-tokens

### 2.5 Pipeline: text-generation com GPT-2 português

- [ ] `gerador = pipeline("text-generation", model="pierreguillou/gpt2-small-portuguese")`
- [ ] Prompt: `"Interação entre Amoxicilina e Ibuprofeno:"`
- [ ] Mostrar saída (provável alucinação)
- [ ] Variar parâmetros: `temperature=0.7`, `top_k=50`, `max_length=100`
- [ ] Célula Markdown: decoder-only (GPT-2) vs encoder-only (BERT), alucinação, parâmetros de geração

### 2.6 Pipeline: fill-mask

- [ ] `unmasker = pipeline("fill-mask", model="neuralmind/bert-base-portuguese-cased")`
- [ ] Testar: `"O uso concomitante é [MASK] em casos de insuficiência renal"`
- [ ] `top_k=5` — analisar tokens previstos
- [ ] Célula Markdown: fill-mask revela conhecimento latente do BERT sobre relações entre palavras

### 2.7 Pipeline: summarization

- [ ] `summarizer = pipeline("summarization", model="facebook/bart-large-cnn")`
- [ ] Testar com seção de interações de uma bula real (Fonte 2, texto ~500 palavras)
- [ ] `max_length=80, min_length=30, do_sample=False`
- [ ] Célula Markdown: summarization abstrativa vs extrativa, desafios em português, BART como encoder-decoder

### 2.8 Pipeline: question-answering

- [ ] `qa = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")`
- [ ] Contexto: parágrafo da Fonte 1 sobre interações com Metotrexato
- [ ] Pergunta: `"Quais medicamentos interagem com Amoxicilina?"`
- [ ] Pergunta: `"Qual o risco da interação com Metotrexato?"`
- [ ] Célula Markdown: QA extrativo — só funciona se resposta está literalmente no texto; limitação para perguntas inferenciais

### 2.9 Tabela comparativa de modelos

- [ ] Tabela Markdown:

| Modelo | Arquitetura | Parâmetros | Tarefa | Limite Tokens | Domínio |
|---|---|---|---|---|---|
| `clinicalnerpt-chemical` | BERT (encoder-only) | 110M | NER | 512 | Clínico PT |
| `biobertpt-all` | BERT (encoder-only) | 110M | Classificação | 512 | Biomédico PT |
| `bart-large-cnn` | BART (encoder-decoder) | 406M | Summarization | 1024 | Genérico EN |
| `gpt2-small-portuguese` | GPT-2 (decoder-only) | 124M | Text Generation | 1024 | Genérico PT |
| `bert-base-portuguese-cased` | BERT (encoder-only) | 110M | Fill-mask / Embeddings | 512 | Genérico PT |

- [ ] Célula Markdown: encoder-only (compreensão, classificação, NER) vs decoder-only (geração) vs encoder-decoder (tradução, sumarização). Pipeline vs inferência manual. Limite de 512 tokens do BERT e como contornar (chunking).

### 2.10 Conclusão

- [ ] Resumo: quais tarefas são úteis para o detector de interações
  - NER → extrair medicamentos da consulta e das bulas
  - Classificação → classificar interações (com fine-tuning)
  - QA → extrair evidências textuais
  - Summarization → condensar seções longas (bônus)
- [ ] Célula Markdown: próximos passos — anotação do dataset para fine-tuning

---

## Artefatos Produzidos

```
c01_modelos_llm.ipynb
```

---

## Verificação

- [ ] Notebook executa do início ao fim sem erros (Kernel → Restart & Run All)
- [ ] `torch.cuda.is_available()` → `True` em todas as células que usam GPU
- [ ] Pipeline NER extrai pelo menos 2 entidades do trecho de teste
- [ ] Tabela comparativa preenchida e explicada nas células Markdown
- [ ] Células Markdown explicam cada pipeline no contexto do projeto
- [ ] Commit: `git add c01_modelos_llm.ipynb && git commit -m "feat: Fase 2 — Notebook 01: modelos NLP com Hugging Face"`
