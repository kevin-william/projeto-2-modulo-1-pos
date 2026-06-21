"""Build complete c01_modelos_llm.ipynb — all cells."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.6"},
}
C = []

# ── Title ──
C.append(nbf.v4.new_markdown_cell(
    "# Notebook 01 — Modelos LLM e NLP com Hugging Face\n\n"
    "**Objetivo:** Demonstrar dominio do ecossistema Hugging Face com tarefas NLP "
    "aplicadas ao dominio de bulas medicas.\n\n"
    "**Rubrica 1:** Construir aplicacoes NLP com LLMs e ecossistema Hugging Face (5 itens)."
))

# ── 2.1 Setup ──
C.append(nbf.v4.new_markdown_cell("## 2.1 Setup e Imports"))
C.append(nbf.v4.new_code_cell(
    "import torch\n"
    "from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForQuestionAnswering\n"
    "from transformers import AutoModelForSeq2SeqLM\n"
    "from scripts.config import DEVICE, NER_MODEL, EMBEDDING_MODEL\n\n"
    'print(f"PyTorch: {torch.__version__}")\n'
    'print(f"CUDA disponivel: {torch.cuda.is_available()}")\n'
    'print(f"Device configurado: {DEVICE}")\n'
    "if torch.cuda.is_available():\n"
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
    '    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'
))

# ── 2.2 AutoModel ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.2 Carregando Modelo com AutoModel + AutoTokenizer\n\n"
    "Demonstracao no estilo do professor: `AutoTokenizer`, `AutoModel`, inspecao de hidden states.\n\n"
    "**Modelo:** `pucpr/clinicalnerpt-chemical` — BERT para NER clinico em portugues."
))
C.append(nbf.v4.new_code_cell(
    'model_id = NER_MODEL\n'
    "tokenizer = AutoTokenizer.from_pretrained(model_id)\n"
    "model = AutoModel.from_pretrained(model_id).to(DEVICE)\n\n"
    'inputs = tokenizer("O mecanismo de atencao e poderoso", return_tensors="pt")\n'
    "inputs = {k: v.to(DEVICE) for k, v in inputs.items()}\n"
    "outputs = model(**inputs)\n\n"
    'print(f"Dimensoes do output: {outputs.last_hidden_state.shape}")\n'
    'print(f"Interpretacao: [Batch={outputs.last_hidden_state.shape[0]}, '
    'Tokens={outputs.last_hidden_state.shape[1]}, '
    'Hidden_Dim={outputs.last_hidden_state.shape[2]}]")\n\n'
    'tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])\n'
    'print(f"\\nTokens: {tokens}")\n'
    'print(f"Total de tokens: {len(tokens)}")'
))
C.append(nbf.v4.new_markdown_cell(
    "**Observacoes:**\n- Tokenizador BERT usa **WordPiece**: palavras frequentes → tokens unicos, "
    "raras → sub-tokens\n- Limite: **512 tokens**\n- `last_hidden_state`: embedding contextualizado "
    "de cada token"
))

# ── 2.3 sentiment-analysis ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.3 Pipeline: sentiment-analysis em Frases Clinicas\n\n"
    "Modelo generico em dominio especializado — demonstrando limitacoes que motivam fine-tuning."
))
C.append(nbf.v4.new_code_cell(
    'classifier = pipeline("sentiment-analysis")\n\n'
    'frases = [\n'
    '    "O uso concomitante e contraindicado devido ao risco de arritmia fatal.",\n'
    '    "Nao ha interacoes conhecidas com este medicamento.",\n'
    '    "Recomenda-se monitoramento da funcao renal durante o tratamento.",\n'
    '    "A administracao concomitante de Amoxicilina com Metotrexato pode aumentar a toxicidade.",\n'
    '    "O medicamento e seguro e bem tolerado pela maioria dos pacientes.",\n'
    ']\n\n'
    'for frase in frases:\n'
    '    resultado = classifier(frase)[0]\n'
    '    print(f"[{resultado[\"label\"]:>8} | {resultado[\"score\"]:.3f}] {frase}\")'
))
C.append(nbf.v4.new_markdown_cell(
    "**Analise:** O modelo classifica por tom emocional, nao por significado clinico. "
    "Precisamos de modelos treinados em dominio clinico."
))

# ── 2.4 NER ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.4 Pipeline: NER com clinicalnerpt-chemical\n\n"
    "**NER (token classification):** identifica nomes de medicamentos — principios ativos e nomes comerciais."
))
C.append(nbf.v4.new_code_cell(
    'ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple",\n'
    '             device=0 if DEVICE == "cuda" else -1)\n\n'
    'trecho = (\n'
    '    "A probenecida reduz a secrecao tubular renal da amoxicilina. "\n'
    '    "No uso concomitante com amoxicilina, pode haver aumento dos niveis "\n'
    '    "de amoxicilina no sangue. A administracao concomitante de alopurinol "\n'
    '    "durante o tratamento com amoxicilina pode aumentar a probabilidade "\n'
    '    "de reacoes alergicas da pele. Existem casos raros de INR aumentada "\n'
    '    "em pacientes mantidos com acenocumarol ou varfarina."\n'
    ')\n\n'
    'entidades = ner(trecho)\n'
    'print("Entidades encontradas:")\n'
    'for ent in entidades:\n'
    '    print(f\'{ent["word"]:<25} {ent["score"]:>8.3f} [{ent["start"]}:{ent["end"]}]\')\n\n'
    'unicos = list(set(ent["word"] for ent in entidades))\n'
    'print(f"\\nMedicamentos identificados ({len(unicos)}): {unicos}")'
))
C.append(nbf.v4.new_markdown_cell(
    "**Analise:** Modelo identifica corretamente amoxicilina, probenecida, alopurinol, "
    "acenocumarol, varfarina. Encoder-only (BERT) ideal para NER. Este sera o primeiro "
    "estagio do pipeline RAG."
))

# ── 2.5 text-generation ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.5 Pipeline: text-generation com GPT-2 Portugues\n\n"
    "**Decoder-only:** geracao autoregressiva token por token. Demonstracao de alucinacao."
))
C.append(nbf.v4.new_code_cell(
    'gerador = pipeline("text-generation", model="pierreguillou/gpt2-small-portuguese")\n\n'
    'prompt = "Interacao entre Amoxicilina e Ibuprofeno:"\n\n'
    'r1 = gerador(prompt, max_length=80, do_sample=True, temperature=0.9)\n'
    'print("=== temperature=0.9 (criativa) ===")\n'
    'print(r1[0]["generated_text"])\n\n'
    'r2 = gerador(prompt, max_length=80, do_sample=True, temperature=0.3, top_k=20)\n'
    'print("\\n=== temperature=0.3 + top_k=20 (conservadora) ===")\n'
    'print(r2[0]["generated_text"])'
))
C.append(nbf.v4.new_markdown_cell(
    "**Analise:** GPT-2 gera texto fluente mas **alucina** informacoes — nao tem conhecimento "
    "medico real. Decoder-only (geracao) vs Encoder-only (compreensao)."
))

# ── 2.6 fill-mask ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.6 Pipeline: fill-mask com BERT Portugues\n\n"
    "Revela conhecimento latente do modelo ao prever tokens mascarados."
))
C.append(nbf.v4.new_code_cell(
    'unmasker = pipeline("fill-mask", model=EMBEDDING_MODEL)\n\n'
    'print("=== Teste 1: contexto clinico ===")\n'
    'r1 = unmasker("O uso concomitante de Amoxicilina com Metotrexato e [MASK] devido ao risco de toxicidade.")\n'
    'for r in r1:\n'
    '    print(f"  {r[\"token_str\"]:>12} | score={r[\"score\"]:.4f}")\n\n'
    'print("\\n=== Teste 2: contexto de seguranca ===")\n'
    'r2 = unmasker("Nao ha interacoes conhecidas. O medicamento e [MASK] para uso.")\n'
    'for r in r2[:3]:\n'
    '    print(f"  {r[\"token_str\"]:>12} | score={r[\"score\"]:.4f}")'
))
C.append(nbf.v4.new_markdown_cell(
    "**Analise:** BERT preve 'contraindicado' no contexto de toxicidade, 'seguro' no contexto "
    "positivo. Atencao **bidirecional** permite preencher lacunas. Este modelo sera usado "
    "para gerar embeddings na Fase 6."
))

# ── 2.7 summarization ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.7 Sumarizacao com BART (via AutoModel)\n\n"
    "**Encoder-decoder:** sumarizacao abstrativa. A partir do Transformers 5.x, "
    "usamos `AutoModelForSeq2SeqLM` diretamente."
))
C.append(nbf.v4.new_code_cell(
    'summ_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")\n'
    'summ_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(DEVICE)\n\n'
    'texto = (\n'
    '    "Miopatia pode ocorrer em pacientes que usam Zarator, sendo mais frequentes "\n'
    '    "naqueles que usam tambem ciclosporina, fibratos, niacina ou antifungicos "\n'
    '    "azolicos. A administracao concomitante com medicamentos inibidores do "\n'
    '    "citocromo P450 3A4 (ciclosporina, eritromicina/claritromicina, inibidores "\n'
    '    "da protease) pode alterar a quantidade de atorvastatina no sangue. Sao "\n'
    '    "conhecidas interacoes com antiacidos, colestipol, contraceptivos orais, "\n'
    '    "varfarina, acido fusidico."\n'
    ')\n\n'
    'inputs = summ_tokenizer(texto, max_length=1024, truncation=True, return_tensors="pt").to(DEVICE)\n'
    'summary_ids = summ_model.generate(inputs["input_ids"], max_length=80, min_length=30,\n'
    '                                   num_beams=4, early_stopping=True)\n'
    'resumo = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)\n'
    'print(f"Original ({len(texto.split())} palavras):")\n'
    'print(texto[:200] + "...")\n'
    'print(f"\\nResumo ({len(resumo.split())} palavras):")\n'
    'print(resumo)'
))
C.append(nbf.v4.new_markdown_cell(
    "**Analise:** BART e ~4x maior que BERT (406M vs 110M). Treinado em ingles → qualidade "
    "limitada em portugues. `model.generate()` da controle sobre beams, early stopping."
))

# ── 2.8 QA ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.8 Question Answering com BERT em Portugues (via AutoModel)\n\n"
    "**QA extrativo:** encontra span de resposta no contexto. "
    "A partir do Transformers 5.x, usamos `AutoModelForQuestionAnswering` diretamente."
))
C.append(nbf.v4.new_code_cell(
    'qa_model_id = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"\n'
    'qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_id)\n'
    'qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_id).to(DEVICE)\n\n'
    'contexto = (\n'
    '    "A probenecida reduz a secrecao tubular renal da amoxicilina. "\n'
    '    "A administracao concomitante de alopurinol durante o tratamento "\n'
    '    "com amoxicilina pode aumentar a probabilidade de reacoes alergicas "\n'
    '    "da pele. Existem casos raros de INR aumentada em pacientes mantidos "\n'
    '    "com acenocumarol ou varfarina, ao receberem um curso de tratamento "\n'
    '    "com amoxicilina. Se a coadministracao e necessaria, o tempo de "\n'
    '    "protrombina ou INR deve ser cuidadosamente monitorado."\n'
    ')\n\n'
    'perguntas = [\n'
    '    "Quais medicamentos interagem com Amoxicilina?",\n'
    '    "Qual o risco de tomar Amoxicilina com Varfarina?",\n'
    '    "Qual a dose maxima recomendada de Amoxicilina?",\n'
    ']\n\n'
    'for i, pergunta in enumerate(perguntas, 1):\n'
    '    inputs = qa_tokenizer(pergunta, contexto, max_length=512,\n'
    '                          truncation=True, return_tensors="pt").to(DEVICE)\n'
    '    outputs = qa_model(**inputs)\n'
    '    start_idx = outputs.start_logits.argmax()\n'
    '    end_idx = outputs.end_logits.argmax()\n'
    '    answer_ids = inputs["input_ids"][0][start_idx:end_idx+1]\n'
    '    answer = qa_tokenizer.decode(answer_ids)\n'
    '    score = (outputs.start_logits.max() + outputs.end_logits.max()).item()\n'
    '    print(f"P{i}: {answer} (score: {score:.1f})")'
))
C.append(nbf.v4.new_markdown_cell(
    "**Analise:** P1 extrai medicamentos corretamente. P2 encontra 'INR aumentada' "
    "(associacao indireta). P3 retorna algo com score baixo — QA extrativo **sempre "
    "retorna um span**, mesmo sem resposta. **Licao para o RAG:** usaremos geracao "
    "fundamentada (LLM + contexto), que pode dizer 'sem informacao'."
))

# ── 2.9 Tabela comparativa ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.9 Tabela Comparativa de Modelos e Arquiteturas\n\n"
    "| Modelo | Arquitetura | Param. | Tarefa | Limite | Dominio |\n"
    "|---|---|---|---|---|---|\n"
    "| `clinicalnerpt-chemical` | BERT (encoder-only) | 110M | NER | 512 | Clinico PT |\n"
    "| `biobertpt-all` | BERT (encoder-only) | 110M | Classificacao | 512 | Biomedico PT |\n"
    "| `bart-large-cnn` | BART (encoder-decoder) | 406M | Sumarizacao | 1024 | Generico EN |\n"
    "| `gpt2-small-portuguese` | GPT-2 (decoder-only) | 124M | Geracao | 1024 | Generico PT |\n"
    "| `bert-base-portuguese-cased` | BERT (encoder-only) | 110M | Fill-mask / Embeddings | 512 | Generico PT |\n\n"
    "### Diferenças entre Arquiteturas\n\n"
    "- **Encoder-only (BERT):** Atencao bidirecional — cada token ve contexto completo. "
    "Ideal para compreensao: NER, classificacao, QA extrativo, embeddings.\n"
    "- **Decoder-only (GPT-2):** Atencao unidirecional/causal — cada token so ve contexto "
    "anterior. Ideal para geracao: chatbots, completamento de texto.\n"
    "- **Encoder-decoder (BART):** Combina ambos — encoder processa entrada, decoder gera "
    "saida. Ideal para traducao e sumarizacao.\n\n"
    "### Pipeline vs Inferencia Manual\n\n"
    "- `pipeline()`: rapido, encapsula tokenizacao + modelo + post-processamento. "
    "Ideal para prototipagem.\n"
    "- Inferencia manual (`AutoModel` + `tokenizer`): controle total sobre tokenizacao, "
    "GPU, batched inference. Necessario para fine-tuning e producao.\n"
    "- A partir do Transformers 5.x, alguns pipelines foram descontinuados "
    "(ex: `summarization`, `question-answering`) — usar `AutoModel` diretamente."
))

# ── 2.10 Conclusao ──
C.append(nbf.v4.new_markdown_cell(
    "## 2.10 Conclusao: O que Aprendemos\n\n"
    "### Tarefas uteis para o detector de interacoes:\n\n"
    "| Tarefa | Aplicacao no Projeto | Fase |\n"
    "|---|---|---|\n"
    "| **NER** | Extrair medicamentos da consulta do usuario | Fase 8 (RAG) |\n"
    "| **Classificacao** | Classificar interacao (0/1/2) com BioBERTpt fine-tuned | Fase 4 |\n"
    "| **Embeddings** | Indexar chunks no ChromaDB para busca vetorial | Fase 6 |\n"
    "| **Geracao (LLM)** | Produzir resposta final fundamentada nos chunks | Fase 8 (RAG) |\n\n"
    "### Proximos passos:\n"
    "- **Fase 3:** Anotar dataset de treino (~1.500 pares) com weak supervision\n"
    "- **Fase 4:** Fine-tuning do BioBERTpt para classificacao de interacoes\n"
    "- **Fase 5:** Prompt engineering com LLMs (zero-shot, few-shot, CoT)"
))

nb.cells = C
nbf.write(nb, "C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb")
print(f"Notebook rebuilt: {len(C)} cells")
