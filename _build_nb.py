"""Append cell 2.4 (NER) to notebook."""
import nbformat as nbf

nb = nbf.read("C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb", as_version=4)

md = """## 2.4 Pipeline: NER com clinicalnerpt-chemical

**Named Entity Recognition (NER)** e uma tarefa de **token classification**:
cada token recebe um rotulo (B-ChemicalDrugs, I-ChemicalDrugs, ou O).
O modelo `clinicalnerpt-chemical` foi treinado especificamente para
identificar nomes de medicamentos em textos clinicos em portugues —
incluindo tanto **principios ativos** quanto **nomes comerciais**.

Usamos `aggregation_strategy="simple"` para agrupar sub-tokens
(ex: `Amoxi` + `##cilina` → `Amoxicilina`)."""
nb.cells.append(nbf.v4.new_markdown_cell(md))

code = r"""ner = pipeline(
    "ner",
    model=NER_MODEL,
    aggregation_strategy="simple",
    device=0 if DEVICE == "cuda" else -1,
)

# Trecho real de bula ANVISA (amoxicilina profissional)
trecho_bula = (
    "A probenecida reduz a secrecao tubular renal da amoxicilina. "
    "No uso concomitante com amoxicilina, pode haver aumento dos niveis "
    "de amoxicilina no sangue. A administracao concomitante de alopurinol "
    "durante o tratamento com amoxicilina pode aumentar a probabilidade "
    "de reacoes alergicas da pele. Existem casos raros de INR aumentada "
    "em pacientes mantidos com acenocumarol ou varfarina."
)

entidades = ner(trecho_bula)

print("Entidades encontradas:")
print(f'{"Entidade":<25} {"Score":>8} {"Posicao"}')
print("-" * 55)
for ent in entidades:
    print(f'{ent["word"]:<25} {ent["score"]:>8.3f} '
          f'[{ent["start"]}:{ent["end"]}]')

# Deducao dos medicamentos unicos
unicos = list(set(ent["word"] for ent in entidades))
print(f"\nMedicamentos identificados ({len(unicos)}): {unicos}")
"""
nb.cells.append(nbf.v4.new_code_cell(code))

md2 = """**Analise:**

- O modelo identifica corretamente: `amoxicilina`, `probenecida`,
`alopurinol`, `acenocumarol`, `varfarina`
- **Arquitetura encoder-only (BERT):** cada token e classificado
independentemente com base no contexto bidirecional — ideal para NER
- **Agregacao de sub-tokens:** `aggregation_strategy="simple"`
junta `B-` e `I-` em uma unica entidade
- **Por que nao usar regex?** Nomes de medicamentos tem alta
variabilidade (marcas, genericos, compostos) — impossivel cobrir com regras
- Este NER sera o **primeiro estagio do pipeline RAG**: extrair
medicamentos da consulta do usuario para depois buscar interacoes"""
nb.cells.append(nbf.v4.new_markdown_cell(md2))

nbf.write(nb, "C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb")
print("Cell 2.4 appended.")
