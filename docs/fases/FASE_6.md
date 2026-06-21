# Fase 6: Notebook 03 — Embeddings Semânticos e Busca Vetorial

**Objetivo:** Gerar embeddings dos chunks das bulas, indexar no ChromaDB, e comparar estratégias de busca (cosseno puro vs híbrida BM25) com 3 modelos de embeddings.

**Dependências:** Fase 1 (chunks JSONL disponível).

**Rubricas cobertas:** Rubrica 3 — todos os 5 itens.

---

## Estrutura do Notebook

```
c03_embeddings_busca.ipynb
├── 6.1  Setup e imports
├── 6.2  Carregar e inspecionar chunks
├── 6.3  Indexação no ChromaDB
├── 6.4  Busca semântica (cosseno)
├── 6.5  Busca híbrida (BM25 + embeddings)
├── 6.6  Comparação de 3 modelos de embeddings
├── 6.7  Queries de teste e análise de falhas
└── 6.8  Justificativa da estratégia
```

---

## Tarefas

### 6.1 Setup e imports

- [ ] `import chromadb`, `from sentence_transformers import SentenceTransformer`
- [ ] `from rank_bm25 import BM25Okapi`
- [ ] `from scripts.config import *`
- [ ] Verificar GPU para embeddings: `model.to("cuda")`

### 6.2 Carregar chunks

- [ ] Carregar `data/chunks_bulas.jsonl` com pandas
- [ ] Estatísticas: total de chunks, distribuição por fonte, média de tokens
- [ ] Exibir 5 exemplos aleatórios

### 6.3 Indexação no ChromaDB

- [ ] Criar `chromadb.PersistentClient(path=str(CHROMA_DB_DIR))`
- [ ] Criar coleção `bulas_interacoes`:
  ```python
  collection = client.create_collection(
      name="bulas_interacoes",
      metadata={"hnsw:space": "cosine"}
  )
  ```
- [ ] Função `indexar_chunks(chunks_df, modelo, batch_size=32)`:
  - Gerar embeddings em batches
  - Inserir no ChromaDB com metadados: `medicamento`, `fonte`, `secao`
  - ID único: `hash do texto`
  - Barra de progresso
- [ ] Indexar todos os chunks (~30-50k)
- [ ] Célula Markdown: tamanho da coleção, tempo de indexação, uso de disco

### 6.4 Busca semântica (cosseno)

- [ ] Função `buscar_semantica(query, top_k=5)`:
  ```python
  def buscar_semantica(query: str, top_k: int = 5) -> list[dict]:
      query_embedding = modelo.encode([query])[0].tolist()
      resultados = collection.query(
          query_embeddings=[query_embedding],
          n_results=top_k,
          include=["documents", "metadatas", "distances"]
      )
      # Formatar resultado
      return [...]
  ```
- [ ] Testar: `buscar_semantica("Amoxicilina com Ibuprofeno")` → top-5 resultados
- [ ] Exibir: texto do chunk, medicamento, fonte, distância cosseno

### 6.5 Busca híbrida (BM25 + embeddings)

- [ ] Construir índice BM25 com todos os chunks
  ```python
  from rank_bm25 import BM25Okapi
  tokenized_chunks = [chunk.split() for chunk in textos]
  bm25 = BM25Okapi(tokenized_chunks)
  ```
- [ ] Função `buscar_hibrida(query, top_k=5, alpha=0.5)`:
  - Normalizar scores BM25 para [0, 1]
  - Normalizar scores cosseno para [0, 1] (1 - distance)
  - Score combinado: `alpha * cos_score + (1-alpha) * bm25_score`
  - Reordenar e retornar top-k
- [ ] Comparar top-5 resultados de busca pura vs híbrida para 3 queries

### 6.6 Comparação de 3 modelos de embeddings

- [ ] Testar com 50 queries de validação (pares medicamentosos conhecidos):
  1. `neuralmind/bert-base-portuguese-cased` (BERT pt, 768d)
  2. `intfloat/multilingual-e5-base` (E5 multilíngue, 768d)
  3. `sentence-transformers/all-MiniLM-L6-v2` (MiniLM, 384d, rápido)
- [ ] Métricas para cada modelo:
  - Precision@3, Precision@5
  - MRR (Mean Reciprocal Rank)
  - Tempo de encoding por query
  - Tamanho do embedding
- [ ] Tabela comparativa + gráfico de barras
- [ ] Célula Markdown: trade-off qualidade vs velocidade; E5 multilíngue pode ser melhor para termos técnicos; BERT pt é o nativo

### 6.7 Queries de teste e análise de falhas

- [ ] 10 queries representativas:
  1. `"Amoxicilina com Ibuprofeno"` (interação comum)
  2. `"Dipirona e AAS"` (nomes comerciais)
  3. `"Losartana com Captopril"` (anti-hipertensivos)
  4. `"Metformina e Álcool"` (entidade não-medicamento)
  5. `"AAS Protect com Varfarina"` (nome comercial + princípio ativo)
  6. `"Paracetamol e nada"` (sem interação conhecida)
  7. `"Omeprazol com Clopidogrel"` (interação documentada)
  8. `"Sinvastatina com Cetoconazol"` (interação grave)
  9. `"Dipirona com Paracetamol com Ibuprofeno"` (3 medicamentos)
  10. `"Invexermectina"` (medicamento inexistente)
- [ ] Para cada query: exibir top-3 resultados, avaliar relevância
- [ ] **5 casos de acerto**: query → top-1 resultado contém a resposta correta
- [ ] **5 casos de falha**: query → resultados irrelevantes, explicar por quê:
  - Sinônimos não capturados (ex: "AAS" vs "ácido acetilsalicílico")
  - Contexto muito curto (chunk sem informação suficiente)
  - Entidade rara (medicamento pouco frequente nas bulas)

### 6.8 Justificativa da estratégia

- [ ] Célula Markdown: por que ChromaDB + cosseno?
  - **ChromaDB**: persistente, sem servidor externo, API Python nativa, metadados flexíveis
  - **Cosseno**: adequado para embeddings normalizados (direção > magnitude)
  - **Top-k=5**: balanceia cobertura (encontrar o chunk certo) vs ruído (chunks irrelevantes)
  - **Híbrida BM25**: melhora recall para termos exatos (nomes de medicamentos), mas adiciona latência
- [ ] Recomendação final: usar BERT pt + cosseno + híbrida (alpha=0.3) para o pipeline RAG

---

## Artefatos Produzidos

```
c03_embeddings_busca.ipynb
scripts/embeddings.py
data/chroma_db/   (coleção indexada)
```

---

## Verificação

- [ ] Notebook executa do início ao fim sem erros
- [ ] ChromaDB coleção criada e populada (> 10.000 documentos)
- [ ] `buscar_semantica("Amoxicilina")` retorna ≥ 1 resultado
- [ ] 3 modelos comparados com métricas objetivas
- [ ] 5 acertos + 5 falhas documentados com explicação
- [ ] Commit: `git add c03_embeddings_busca.ipynb scripts/embeddings.py && git commit -m "feat: Fase 6 — Notebook 03: embeddings + ChromaDB + busca híbrida"`
