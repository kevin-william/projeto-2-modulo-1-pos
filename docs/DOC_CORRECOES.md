# Correções — Fase 2: Debug dos Notebooks

## Visão Geral

Este documento registra os problemas encontrados em cada notebook durante a fase de
debug via `scripts/debug_c0X.py`, as causas raiz identificadas e as correções
aplicadas.

---

## c03_embeddings_busca.ipynb

### Bugs Encontrados

#### Bug 1 — `PERSIST_DIR` inexistente
**Sintoma:** `NameError: name 'PERSIST_DIR' is not defined`

**Causa:** `config.py` define `CHROMA_DIR`, não `PERSIST_DIR`. As células do
notebook importavam a variável errada.

**Decisão:** Remover `PERSIST_DIR` do import. Usar `CHROMA_DIR` que já existia.

**Arquivo:** `scripts/debug_c03.py`

```diff
- from scripts.config import (CHUNKS_BULAS, COLLECTION_NAME, CHROMA_DIR, PERSIST_DIR)
+ from scripts.config import (CHUNKS_BULAS, COLLECTION_NAME, CHROMA_DIR)
```

---

#### Bug 2 — `MODELS["MINILM"]` em vez de `"MINILM"`
**Sintoma:** `NameError: name 'sentence-transformers/all-MiniLM-L6-v2' is not defined`

**Causa:** `MODELS` é um dict `{"MINILM": "sentence-transformers/all-MiniLM-L6-v2", ...}`.
O código fazia `MODELS["MINILM"]` (que retorna o path string) e passava isso como
`modelo_nome` para `SentenceTransformer`, que interpretava o path como um path local.

**Decisão:** Passar a key string `"MINILM"` diretamente para `gerar_embeddings()`
e `construir_index()`, não o valor do dict. O wrapper interno é que faz
`MODELS[modelo_nome]`.

**Arquivos afetados:**
- `scripts/debug_c03.py` — cell_09, cell_11, cell_13, cell_16, cell_18

---

#### Bug 3 — Atributo inexistente em `__init__`
**Sintoma:** `AttributeError: 'NoneType' object has no attribute 'query'`

**Causa:** `cell_13` depende de `self.collection` criado por `cell_11`. Quando
`cell_13` roda isoladamente (via `python debug_c03.py 13`), `cell_11` não executou
e `self.collection` é `None`.

**Decisão:** Adicionar carga lazy da collection no início de `cell_13`:

```python
if self.collection is None:
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    self.collection = client.get_collection(COLLECTION_NAME)
```

---

#### Bug 4 — `busca_hibrida()` assinatura errada
**Sintoma:** `TypeError: busca_hibrida() got an unexpected keyword argument 'modelo_nome'`

**Causa:** `busca_hibrida()` recebe `(collection, consulta_texto, embedding, n, alpha)` —
**não** `modelo_nome`. O código das células 16 e 18 passava `modelo_nome=` e
`embedding` separado.

**Decisão:** Corrigir chamada para passar o embedding já calculado como 3º argumento:

```python
# ANTES (errado)
resultados = busca_hibrida(self.collection, QUERY, modelo_nome=modelo, n=5, alpha=0.7)

# DEPOIS (correto)
emb = model.encode([QUERY], normalize_embeddings=True)[0]
resultados = busca_hibrida(self.collection, QUERY, emb, n=5, alpha=0.7)
```

**Arquivos:** `scripts/debug_c03.py` — cells 16 e 18.

---

#### Bug 5 — `CONSULTAS_TESTE` com IDs fake
**Sintoma:** `P@3=0.0` — retrieval retornava zero acertos

**Causa raiz 1 — IDs fictícios:**
As `CONSULTAS_TESTE` usavam IDs como `cvar_087`, `sin_099` que eram de um dataset
sintético/DNER. O ChromaDB real tem IDs como `f1_100290226_etoricoxibe_profissional_000`.

**Decisão:** Substituir IDs fictícios por IDs reais extraídos do `chunks_bulas.jsonl`.
IDs no ChromaDB são normalizados para lowercase (a deduplicação de `construir_index`
aplica `.lower()` nos IDs), então os IDs no documento de ground truth também precisam
estar em lowercase.

```python
# ANTES (fictício)
("sinvastatina itraconazol contraindicado", ["cvar_087", "sin_099"])

# DEPOIS (real, lowercase)
("sinvastatina itraconazol interação contraindicado cyp3a4",
 ["f1_100470472_sinvastatina_profissional_005",
  "f1_100431188_itraconazol_profissional_003"]),
```

**Causa raiz 2 — Ground truth sem menção mútua:**
Vários chunks selecionados para ground truth não continham menção ao segundo
medicamento da query. Ex: chunk `sinvastatina_profissional_003` fala de "CYP3A4"
sem mencionar itraconazol. Chunk correto é `sinvastatina_profissional_005`
que contém "itraconazol" explicitamente.

**Decisão:** Para cada par de consulta, garantir que o chunk de ground truth
contenha menção a AMBOS os medicamentos ou ao conceito relevante (ex: "insuficiência
renal" para metformina).

**Causa raiz 3 — 6 de 10 IDs não existem no ChromaDB:**
Mesmo buscando IDs reais do JSONL, 6 dos 10 IDs não foram encontrados na
collection. Isso ocorre porque `construir_index` remove duplicatas por ID antes de
indexar, e se o mesmo `id` já foi processado com outro conteúdo (deduplicação
agressiva), o chunk pode ter sido descartado.

**IDs faltando no ChromaDB:**
```
MISSING: f1_100431004_amoxicilina_clavulanato...009
MISSING: f1_103700512_varfarina_sódica_paciente_000
MISSING: f1_100430911_losartana_potássica...011
MISSING: f1_102351182_esomeprazol_magnésico...005
MISSING: f1_100431137_atorvastatina_cálcica...000
MISSING: f1_100431331_fosfato_dissódico_de_dexametasona...000
```

**Causa raiz 4 — Retrieval semanticamente falho:**
Para a query "sinvastatina itraconazol", mesmo com o chunk correto
`f1_100470472_sinvastatina_profissional_005` EXISTINDO no ChromaDB (IDs em
lowercase: `f1_100470472_sinvastatina_profissional_005`), o top-5 retornado é:

```
1. f1_109650004_claritromicina_profissional_101   ← errado
2. f1_102350544_claritromicina_profissional_089   ← errado
3. f1_105830943_claritromicina_profissional_089   ← errado
4. f1_100431188_itraconazol_profissional_132       ← contexto genérico
5. f2_rasilez_004                                   ← irrelevante
```

Isso sugere que:
1. O embedding de "sinvastatina itraconazol" é mais similar semanticamente a
   "claritromicina" (talvez por ser um medicamento de referência no dataset bulas)
2. `all-MiniLM-L6-v2` não é otimizado para pharmaco-embeddings em português

**Decisão de arquitetura:** O retrieval atual precisa de pelo menos uma destas
intervenções (para a fase 3 ou 4):
- Fine-tune do embedding model com corpus farmacêutico em português
- BM25 puro (alpha=0.0) como baseline
- Híbrido com peso maior para BM25 (alpha menor)

---

### Métricas Finais (após correções de código)

| Célula | Status | Observação |
|--------|--------|-----------|
| cell_05 | ✅ OK | 155,987 chunks carregados |
| cell_09 | ✅ OK | Embeddings ~25it/s |
| cell_11 | ✅ OK | 155,002 docs indexados (985 dedup) |
| cell_13 | ⚠️ P@3=0.033 | Retrieval semanticamente fraco (ver causa acima) |
| cell_14 | ✅ OK | Gráficos (omitidos no debug — lógica intacta) |
| cell_16 | ⚠️ Timeout | BM25 re-indexing lento com 155k docs |
| cell_18 | ⚠️ Timeout | Sweep alpha (mesma causa) |
| cell_20 | ✅ OK | Estrutura de classes (não executada no debug) |

---

### Correções a Aplicar nos Notebooks

Após validar no `.py`, aplicar as seguintes correções nos notebooks `.ipynb`:

1. **cell_13 e células seguintes:** remover `PERSIST_DIR` do import
2. **cell_09, 11, 13, 16, 18:** corrigir `MODELS["MINILM"]` → `"MINILM"`
3. **cell_16 e 18:** corrigir assinatura de `busca_hibrida()`
4. **cell_13:** adicionar carga lazy da collection
5. **cell_13:** atualizar `CONSULTAS_TESTE` com IDs reais (lowercase)
6. **cell_16 e 18:** otimizar BM25 (pré-computar em cell_11 em vez de re-indexar a cada demo)

---

## Resumo de Decisões Arquiteturais para Fase 3

1. **BM25 como fallback prioritário**: dado que semantic embeddings falham
   para termos farmacológicos em português, BM25 (alpha=0.0) deve ser o baseline.
   Híbrido (alpha=0.3-0.5) pode ajudar para queries com sinônimos.

2. **Ground truth precisa de curadoria manual**: não é possível automatizar a
   construção de ground truth para métricas de retrieval — os chunks precisam
   ser validados manualmente para garantir que o chunk recuperado é realmente
   relevante para a query.

3. **Fine-tuning de embeddings (futuro)**: se o pipeline híbrido não atingir
   P@3 > 0.5, considerar fine-tunar `all-MiniLM-L6-v2` ou similar com o corpus
   de bulas (similarity learning task).
