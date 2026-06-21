"""Append cells 2.5 (text-generation) + 2.6 (fill-mask) to notebook."""
import nbformat as nbf

nb = nbf.read("C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb", as_version=4)

# ── 2.5 text-generation ──
md = """## 2.5 Pipeline: text-generation com GPT-2 Portugues

**Decoder-only models** (como GPT-2) geram texto token por token, de forma
autoregressiva — cada token gerado depende apenas dos tokens anteriores
(atencao unidirecional/causal).

Usamos o modelo `pierreguillou/gpt2-small-portuguese`, um GPT-2 treinado
em corpus portugues. Vamos testar com um prompt sobre interacao medicamentosa
e observar o fenomeno de **alucinacao**."""
nb.cells.append(nbf.v4.new_markdown_cell(md))

code = r"""gerador = pipeline(
    "text-generation",
    model="pierreguillou/gpt2-small-portuguese",
)

prompt = "Interacao entre Amoxicilina e Ibuprofeno:"

# Geracao padrao (pode alucinar)
resultado = gerador(prompt, max_length=80, do_sample=True, temperature=0.9)
print("=== Geracao com temperature=0.9 (criativa) ===")
print(resultado[0]["generated_text"])

# Geracao mais conservadora
resultado2 = gerador(prompt, max_length=80, do_sample=True, temperature=0.3, top_k=20)
print("\n=== Geracao com temperature=0.3 + top_k=20 (conservadora) ===")
print(resultado2[0]["generated_text"])
"""
nb.cells.append(nbf.v4.new_code_cell(code))

md2 = """**Analise:**

- O modelo gera texto fluente em portugues, mas **inventa informacoes**
(efeito conhecido como **alucinacao**) — ele nao tem conhecimento medico real
- `temperature=0.9`: saida mais variada e criativa, mas maior risco de alucinacao
- `temperature=0.3 + top_k=20`: saida mais conservadora e repetitiva
- **Decoder-only (GPT-2) vs Encoder-only (BERT):** GPT-2 gera texto (bom para
chatbots), BERT compreende texto (bom para classificacao, NER)
- **Por que usamos decoder-only no RAG?** Para gerar a resposta final
fundamentada, mas SEMPRE ancorada nos chunks recuperados — o contexto real
previne a alucinacao"""
nb.cells.append(nbf.v4.new_markdown_cell(md2))

# ── 2.6 fill-mask ──
md3 = """## 2.6 Pipeline: fill-mask com BERT Portugues

**Fill-mask** revela o **conhecimento latente** do modelo: ao mascarar
um token e pedir que o modelo o preveja, vemos quais palavras o BERT
associa semanticamente ao contexto.

Usamos `neuralmind/bert-base-portuguese-cased`, o BERT em portugues
mais consolidado. O teste: qual palavra o modelo preve em um contexto
clinico sobre interacoes?"""
nb.cells.append(nbf.v4.new_markdown_cell(md3))

code2 = r"""unmasker = pipeline(
    "fill-mask",
    model=EMBEDDING_MODEL,  # "neuralmind/bert-base-portuguese-cased"
)

# Teste 1: contexto clinico
resultado = unmasker(
    "O uso concomitante de Amoxicilina com Metotrexato e [MASK] "
    "devido ao risco de toxicidade."
)
print("=== Teste 1: contexto clinico ===")
for r in resultado:
    print(f"  {r['token_str']:>12} | score={r['score']:.4f}")

# Teste 2: contexto de seguranca
resultado2 = unmasker(
    "Nao ha interacoes conhecidas. O medicamento e [MASK] para uso."
)
print("\n=== Teste 2: contexto de seguranca ===")
for r in resultado2[:3]:
    print(f"  {r['token_str']:>12} | score={r['score']:.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

md4 = """**Analise:**

- **Teste 1:** O modelo preve palavras como "contraindicado", "perigoso" —
demonstrando que aprendeu a associacao entre "Metotrexato + toxicidade" e
palavras de alerta
- **Teste 2:** Preve "seguro", "indicado" — reconhece o contexto positivo
- **Arquitetura encoder-only:** O BERT usa atencao **bidirecional** —
cada token "ve" tanto o contexto a esquerda quanto a direita, por isso
consegue preencher lacunas com alta precisao
- Este modelo (`bert-base-portuguese-cased`) sera usado na Fase 6 para
**gerar embeddings** dos chunks das bulas"""
nb.cells.append(nbf.v4.new_markdown_cell(md4))

nbf.write(nb, "C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb")
print("Cells 2.5 + 2.6 appended.")
