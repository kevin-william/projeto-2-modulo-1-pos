"""Insere celula de definicao de PARES_VALIDACAO antes da celula de avaliacao."""
import json

with open('c03_embeddings_busca.ipynb') as f:
    nb = json.load(f)

# Texto da nova celula markdown
md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3.6.1 Dataset de Validacao (Ground Truth)\n"]
}

# Texto da nova celula code
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Ground truth: consulta -> farmaco esperado no resultado\n",
        "PARES_VALIDACAO = [\n",
        "    {\"consulta\": \"amoxicilina\", \"medicamento\": \"amoxicilina\"},\n",
        "    {\"consulta\": \"dipirona\", \"medicamento\": \"dipirona\"},\n",
        "    {\"consulta\": \"metformina\", \"medicamento\": \"metformina\"},\n",
        "    {\"consulta\": \"sinvastatina\", \"medicamento\": \"sinvastatina\"},\n",
        "    {\"consulta\": \"omeprazol\", \"medicamento\": \"omeprazol\"},\n",
        "    {\"consulta\": \"losartana\", \"medicamento\": \"losartana\"},\n",
        "    {\"consulta\": \"alopurinol\", \"medicamento\": \"alopurinol\"},\n",
        "    {\"consulta\": \"atenolol\", \"medicamento\": \"atenolol\"},\n",
        "    {\"consulta\": \"ibuprofeno\", \"medicamento\": \"ibuprofeno\"},\n",
        "    {\"consulta\": \"warfaria\", \"medicamento\": \"warfarina\"},   # nome similar\n",
        "    {\"consulta\": \"aspirina\", \"medicamento\": \"acido acetilsalicilico\"},\n",
        "    {\"consulta\": \"paracetamol\", \"medicamento\": \"paracetamol\"},\n",
        "]\n",
        "\n",
        "print(\"Dataset de validacao: %d pares\" % len(PARES_VALIDACAO))\n",
        "for p in PARES_VALIDACAO:\n",
        "    print(\"  Consulta: %-30s  Esperado: %s\" % (p[\"consulta\"], p[\"medicamento\"]))\n"
    ]
}

# Encontrar o indice da celula que usa calcular_recall_at_k (celula 13)
target_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', ''))
    if 'calcular_recall_at_k' in src:
        target_idx = i
        break

if target_idx is None:
    print("Celula com calcular_recall_at_k nao encontrada!")
else:
    print("Inserindo antes da celula %d (target_idx)" % target_idx)
    nb['cells'].insert(target_idx, code_cell)
    nb['cells'].insert(target_idx, md_cell)
    with open('c03_embeddings_busca.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print("Feito. Nova estrutura:")
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', ''))
        first = src.split('\n')[0][:70]
        print("  Cell %d [%s]: %s" % (i, cell['cell_type'], first))
