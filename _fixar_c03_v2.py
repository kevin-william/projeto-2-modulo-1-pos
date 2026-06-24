"""Divide Cell 15 em duas: def calcular_recall_at_k + avaliacao."""
import json

with open('c03_embeddings_busca.ipynb') as f:
    nb = json.load(f)

# Cell 15 atual contem def + avaliacao
cell15 = nb['cells'][15]
src = ''.join(cell15['source'])
print("Cell 15 atual:")
print(src[:300])
print("...\n")

# Separa: tudo ate "def calcular_recall_at_k" twice é a primeira def incompleta
# O segundo "def calcular_recall_at_k" comeca o corpo correto
# A linha "recall, acertos, total = calcular_recall_at_k" comeca a avaliacao

lines = src.split('\n')
# Encontrar onde comeca a segunda definicao
second_def_idx = None
for idx, line in enumerate(lines):
    if 'def calcular_recall_at_k' in line and idx > 0:
        second_def_idx = idx
        break

# Encontrar onde comeca a avaliacao (chamada a calcular_recall_at_k)
eval_start = None
for idx, line in enumerate(lines):
    if 'recall, acertos, total = calcular_recall_at_k' in line:
        eval_start = idx
        break

print(f"second_def_idx={second_def_idx}, eval_start={eval_start}")

def_body = lines[second_def_idx:eval_start]
eval_body = lines[eval_start:]

print("DEF body (%d linhas):" % len(def_body))
print('\n'.join(def_body[:3]), "...")
print("\nEVAL body (%d linhas):" % len(eval_body))
print('\n'.join(eval_body[:3]), "...")

# Nova celula 15: apenas a funcao
cell15_new_source = '\n'.join(def_body) + '\n'
cell15_new = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [cell15_new_source]
}

# Nova celula 16: a avaliacao
eval_source = '\n'.join(eval_body) + '\n'
cell16_new = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [eval_source]
}

# Substitui cell 15 (antiga) por cell 15 (nova def) + cell 16 (avaliacao)
# Encontra posicao de cell 15 no nb
cell15_pos = None
for i, c in enumerate(nb['cells']):
    if c is cell15:
        cell15_pos = i
        break
print(f"\nCell 15 esta na posicao {cell15_pos}")

nb['cells'][cell15_pos] = cell15_new
nb['cells'].insert(cell15_pos + 1, cell16_new)

print("\nNova estrutura:")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', ''))
    first = src.split('\n')[0][:70]
    print(f"  Cell {i} [{c['cell_type']}]: {first}")

with open('c03_embeddings_busca.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("\nNotebook salvo.")
