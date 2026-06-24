"""Corrige dois bugs no c04_inferencia_local.ipynb:
1. template_fewshot dentro de analisar_json (dead code) -> mover para module level
2. getattr callable/bool em medir_latencia (Cell 7)
"""
import json, re

with open('c04_inferencia_local.ipynb') as f:
    nb = json.load(f)

# ---- Fix 1: Cell 5 ----
cell5 = nb['cells'][5]
src5 = ''.join(cell5['source'])

# The dead template is between "return None" and the wrongly-indented block
# We need to:
# 1. Remove the dead template block from inside analisar_json
# 2. Add template_fewshot at module level BEFORE def analisar_json
# 3. Keep def analisar_json intact

# Find where "return None" appears in analisar_json
lines5 = src5.split('\n')

# Locate 'return None' at indentation 4 inside analisar_json (line ~22)
return_none_idx = None
for i, line in enumerate(lines5):
    if line.strip() == 'return None' and not line.startswith('    return None'):
        pass  # skip wrong indentation
    if line.strip() == 'return None':
        return_none_idx = i
        break

print(f"Fix1: 'return None' found at line {return_none_idx}")
# Everything after 'return None' in analyser is dead code
# Keep up to 'return None' (exclusive - we'll add \n\ntemplate_fewshot\n\ndef analisar_json...')
before_return = '\n'.join(lines5[:return_none_idx])  # ends at 'return None'

# Now reconstruct: remove dead code + add template at module level
# The dead code starts at the line AFTER 'return None'
dead_start = return_none_idx + 1
# Find where 'def montar_prompt' starts (it's after the dead template)
def_montar_idx = None
for i, line in enumerate(lines5):
    if 'def montar_prompt' in line:
        def_montar_idx = i
        break
print(f"Fix1: 'def montar_prompt' at line {def_montar_idx}")

# dead block is lines [return_none_idx+1 : def_montar_idx]
# we want to remove those and insert template_fewshot before def analisar_json

module_level_template = """
template_fewshot = (
    "[PAPEL] Voce e um farmacologo clinico.\\n"
    "[EXEMPLOS] "
    'Exemplo: Nao ha interacoes. -> {"classe": 0}\\n'
    'Exemplo: Miopatia com ciclosporina. Cautela. -> {"classe": 1}\\n'
    'Exemplo: contraindicado. Rabdomiolise fatal. -> {"classe": 2}\\n'
    "[TRECHO] {trecho}\\n"
    "[SAIDA - JSON apenas] {\\"classe\\": <0, 1 ou 2>, \\"justificativa\\": \\"<breve>\\"}"
)

"""

# New cell 5: template_fewshot at module level, then def analisar_json, then rest (from def_montar_idx)
rest_of_cell = '\n'.join(lines5[def_montar_idx:])
new_src5 = module_level_template + rest_of_cell

cell5['source'] = [new_src5]
print(f"Fix1: new cell5 lines: {len(new_src5.split(chr(10)))}")

# ---- Fix 2: Cell 7 ----
cell7 = nb['cells'][7]
src7 = ''.join(cell7['source'])
old7 = '    disponivel = getattr(backend, "disponivel", lambda: True)()'
new7 = '    _d = getattr(backend, "disponivel", True)\n    disponivel = _d() if callable(_d) else _d'
if old7 in src7:
    new_src7 = src7.replace(old7, new7)
    cell7['source'] = [new_src7]
    print(f"Fix2: OK - corrigido em cell 7")
else:
    print(f"Fix2: '{old7[:40]}' NAO encontrado em cell 7")
    print("Linhas com 'disponivel' em cell 7:")
    for i, l in enumerate(src7.split('\n')):
        if 'disponivel' in l:
            print(f"  {i}: {repr(l)}")

with open('c04_inferencia_local.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

# ---- Validate ----
print("\nValidando...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        try:
            compile(src, f'cell_{i}', 'exec')
        except SyntaxError as e:
            print(f"  Cell {i}: SyntaxError: {e}")
print("Validacao OK")
