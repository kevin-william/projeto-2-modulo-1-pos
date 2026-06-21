"""Build c01_modelos_llm.ipynb programmatically."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.6"},
}

cells = []

# ── Title ──
cells.append(nbf.v4.new_markdown_cell(
    "# Notebook 01 — Modelos LLM e NLP com Hugging Face\n\n"
    "**Objetivo:** Demonstrar dominio do ecossistema Hugging Face com tarefas NLP "
    "aplicadas ao dominio de bulas medicas, seguindo o estilo do professor "
    "(`pipeline`, `AutoTokenizer`, `AutoModel`).\n\n"
    "**Rubrica 1:** Construir aplicacoes NLP com LLMs e ecossistema Hugging Face (5 itens)."
))

# ── 2.1 Setup ──
cells.append(nbf.v4.new_markdown_cell("## 2.1 Setup e Imports"))
cells.append(nbf.v4.new_code_cell(
    "import torch\n"
    "from transformers import pipeline, AutoModel, AutoTokenizer\n"
    "from scripts.config import DEVICE, NER_MODEL, EMBEDDING_MODEL\n\n"
    'print(f"PyTorch: {torch.__version__}")\n'
    'print(f"CUDA disponivel: {torch.cuda.is_available()}")\n'
    'print(f"Device configurado: {DEVICE}")\n'
    "if torch.cuda.is_available():\n"
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
    '    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'
))

# ── 2.2 AutoModel + AutoTokenizer ──
cells.append(nbf.v4.new_markdown_cell(
    "## 2.2 Carregando Modelo com AutoModel + AutoTokenizer\n\n"
    "Demonstracao no estilo do professor: carregar um modelo pre-treinado, "
    "tokenizar entrada, e inspecionar as dimensoes dos hidden states.\n\n"
    "**Modelo:** `pucpr/clinicalnerpt-chemical` — BERT treinado para NER em "
    "textos clinicos em portugues.\n\n"
    "### Por que comecar com AutoModel?\n\n"
    "`AutoModel.from_pretrained()` carrega o corpo do modelo (encoder) "
    "**sem cabecalho de tarefa** — util para entender a arquitetura antes de "
    "adicionar classificadores. As dimensoes `[Batch, Tokens, Hidden_Dim]` revelam:\n"
    "- **Batch:** quantas frases processadas de uma vez\n"
    "- **Tokens:** quantos tokens a tokenizacao gerou (incluindo `[CLS]` e `[SEP]`)\n"
    "- **Hidden_Dim:** tamanho do embedding interno (768 para BERT base)"
))

cells.append(nbf.v4.new_code_cell(
    'model_id = NER_MODEL  # "pucpr/clinicalnerpt-chemical"\n\n'
    "tokenizer = AutoTokenizer.from_pretrained(model_id)\n"
    "model = AutoModel.from_pretrained(model_id).to(DEVICE)\n\n"
    "# Processando a entrada (estilo do professor)\n"
    'inputs = tokenizer("O mecanismo de atencao e poderoso", return_tensors="pt")\n'
    "# Move para GPU\n"
    "inputs = {k: v.to(DEVICE) for k, v in inputs.items()}\n"
    "outputs = model(**inputs)\n\n"
    'print(f"Dimensoes do output: {outputs.last_hidden_state.shape}")\n'
    'print(f"Interpretacao: [Batch={outputs.last_hidden_state.shape[0]}, '
    'Tokens={outputs.last_hidden_state.shape[1]}, '
    'Hidden_Dim={outputs.last_hidden_state.shape[2]}]")\n\n'
    "# Mostrar os tokens gerados\n"
    'tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])\n'
    'print(f"\\nTokens: {tokens}")\n'
    'print(f"Total de tokens: {len(tokens)}")'
))

cells.append(nbf.v4.new_markdown_cell(
    "**Observacoes:**\n"
    "- O tokenizador BERT usa **WordPiece**: palavras frequentes viram tokens "
    "unicos, palavras raras sao quebradas em sub-tokens\n"
    "- O limite padrao e **512 tokens** — textos mais longos precisam de "
    "estrategias de truncamento ou chunking\n"
    "- `last_hidden_state` contem o embedding contextualizado de cada token — "
    "diferente de embeddings estaticos (Word2Vec), estes variam conforme o "
    "contexto da frase"
))

nb.cells = cells
nbf.write(nb, "C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb")
print("Notebook criado: cells 2.1 + 2.2")
