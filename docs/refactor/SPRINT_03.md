# Sprint 3 — Notebook 03 (Embeddings e Busca Vetorial)

**Objetivo:** Implementar o Notebook 03 demonstrando geração de embeddings com
`sentence-transformers`, indexação no FAISS, busca semântica por cosseno, busca
híbrida com BM25, comparação de dois modelos de embeddings e análise de falhas.

**Duração:** 3-4 horas  
**Commits:** 4 atômicos  
**Rubricas cobertas:** Rubrica 3 (5 itens)

---

## 1. Carregamento de Trechos das Bulas

### 1.1 Função `carregar_trechos_bulas`

```python
def carregar_trechos_bulas(diretorio_raiz, maximo_por_fonte=2500):
    """
    Lê arquivos de bulas diretamente do disco, extrai seções de interações
    medicamentosas e divide em sentenças.

    Para a Fonte 1 (bulas ANVISA): busca seções delimitadas por '##' cujo
    título contenha palavras como 'interação', 'precaução', 'contraindicação'
    ou 'advertência'.

    Para a Fonte 2 (Consultaremedios): busca blocos no formato
    '[P: INTERAÇÃO MEDICAMENTOSA?]' seguidos de 'R:'.

    Cada sentença extraída gera um trecho contendo o nome do medicamento,
    o texto da sentença e a fonte de origem.

    Argumentos:
        diretorio_raiz (Path): Diretório contendo as pastas 'fonte1' e 'fonte2'.
        maximo_por_fonte (int): Limite de arquivos a processar por fonte.

    Retorna:
        list[dict]: Lista de trechos. Cada trecho é um dicionário com:
            - medicamento (str): Nome do medicamento extraído do arquivo
            - texto (str): Sentença extraída da bula
            - fonte (str): 'fonte1' ou 'fonte2'
            - nome_arquivo (str): Nome do arquivo de origem
    """
    trechos = []
    contador_arquivos = 0

    # Padrões de extração
    padrao_secao_fonte1 = re.compile(
        r"##\s*([^\n]+)\s*\n(.*?)(?=\n##|\Z)",
        re.DOTALL
    )
    padrao_bloco_fonte2 = re.compile(
        r"\[P:\s*INTERA[ÇC][AÃ][OO]\s*MEDICAMENTOSA\??\s*\]\s*\nR:\s*(.*?)(?=\n\[P:|\Z)",
        re.DOTALL
    )
    padrao_divisao_sentencas = re.compile(
        r"(?<=[.!?;])\s+(?=[A-ZÀ-Ú\(])"
    )

    # Palavras-chave para identificar seções relevantes na Fonte 1
    palavras_secoes_relevantes = [
        "interaç", "interac", "precauc", "contraind", "advert",
        "devo saber antes", "reaç", "reações"
    ]

    for fonte in ["fonte1", "fonte2"]:
        diretorio_fonte = diretorio_raiz / fonte
        if not diretorio_fonte.is_dir():
            registro.warning("Diretório não encontrado: %s", diretorio_fonte)
            continue

        arquivos = sorted(diretorio_fonte.glob("*.txt"))[:maximo_por_fonte]

        for caminho_arquivo in arquivos:
            contador_arquivos += 1
            nome_arquivo = caminho_arquivo.name

            # Extrair nome do medicamento do nome do arquivo
            nome_medicamento = extrair_nome_medicamento(nome_arquivo)

            try:
                conteudo_arquivo = caminho_arquivo.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                registro.warning("Erro de codificação: %s", nome_arquivo)
                continue

            if fonte == "fonte1":
                # Extrair seções relevantes da Fonte 1
                for resultado_secao in padrao_secao_fonte1.finditer(conteudo_arquivo):
                    nome_secao = resultado_secao.group(1).lower()
                    if not any(p in nome_secao for p in palavras_secoes_relevantes):
                        continue

                    conteudo_secao = resultado_secao.group(2).strip()
                    sentencas = padrao_divisao_sentencas.split(conteudo_secao)

                    for sentenca in sentencas:
                        sentenca = sentenca.strip()
                        if 30 <= len(sentenca) <= 1000:
                            trechos.append({
                                "medicamento": nome_medicamento,
                                "texto": sentenca,
                                "fonte": fonte,
                                "nome_arquivo": nome_arquivo,
                            })
            else:
                # Extrair bloco de interação da Fonte 2
                for resultado_bloco in padrao_bloco_fonte2.finditer(conteudo_arquivo):
                    conteudo_bloco = resultado_bloco.group(1).strip()
                    sentencas = padrao_divisao_sentencas.split(conteudo_bloco)

                    for sentenca in sentencas:
                        sentenca = sentenca.strip()
                        if 30 <= len(sentenca) <= 1000:
                            trechos.append({
                                "medicamento": nome_medicamento,
                                "texto": sentenca,
                                "fonte": fonte,
                                "nome_arquivo": nome_arquivo,
                            })

    registro.info(
        "Trechos carregados: %d (F1: %d, F2: %d) de %d arquivos",
        len(trechos),
        sum(1 for t in trechos if t["fonte"] == "fonte1"),
        sum(1 for t in trechos if t["fonte"] == "fonte2"),
        contador_arquivos,
    )
    return trechos


def extrair_nome_medicamento(nome_arquivo):
    """
    Extrai o nome do medicamento a partir do nome do arquivo.

    Fonte 1: '105830895_amoxicilina_profissional.txt' → 'amoxicilina'
    Fonte 2: 'zarator.txt' → 'zarator'

    Remove prefixo numérico, sufixo de versão e substitui underscores por espaços.
    """
    nome_base = Path(nome_arquivo).stem
    # Remove prefixo numérico (Fonte 1)
    nome_base = re.sub(r"^\d+_", "", nome_base)
    # Remove sufixo de versão
    nome_base = re.sub(r"_(paciente|profissional)$", "", nome_base, flags=re.IGNORECASE)
    # Normaliza: underscores → espaços, minúsculas
    return nome_base.replace("_", " ").strip().lower()
```

---

## 2. Indexação FAISS e Busca Semântica

### 2.1 Geração de embeddings e criação do índice

```python
def criar_indice_busca(trechos, modelo_embeddings):
    """
    Gera embeddings para todos os trechos e cria índice FAISS.

    Os embeddings são normalizados (norma L2 = 1) para que o produto interno
    (Inner Product) do FAISS seja equivalente à similaridade de cosseno.

    Argumentos:
        trechos (list[dict]): Lista de trechos carregados.
        modelo_embeddings (SentenceTransformer): Modelo de embeddings carregado.

    Retorna:
        tuple: (indice_faiss, matriz_embeddings)
            - indice_faiss (faiss.IndexFlatIP): Índice FAISS para busca.
            - matriz_embeddings (np.ndarray): Matriz de embeddings (N, D).
    """
    registro.info("Gerando embeddings para %d trechos...", len(trechos))
    tempo_inicio = __import__("time").time()

    textos = [trecho["texto"] for trecho in trechos]
    dimensao_lote = 32

    matriz_embeddings = modelo_embeddings.encode(
        textos,
        show_progress_bar=True,
        batch_size=dimensao_lote,
        normalize_embeddings=True,  # Normalização L2 → cosseno via IP
    )

    tempo_embeddings = __import__("time").time() - tempo_inicio
    registro.info(
        "Embeddings gerados: %d vetores × %d dimensões em %.1fs",
        matriz_embeddings.shape[0],
        matriz_embeddings.shape[1],
        tempo_embeddings,
    )

    # Criar índice FAISS (Inner Product = cosseno com vetores normalizados)
    dimensao = matriz_embeddings.shape[1]
    indice_faiss = faiss.IndexFlatIP(dimensao)
    indice_faiss.add(matriz_embeddings.astype(np.float32))

    registro.info(
        "Índice FAISS criado: %d vetores indexados",
        indice_faiss.ntotal
    )
    return indice_faiss, matriz_embeddings
```

### 2.2 Função de busca semântica

```python
def buscar_trechos(consulta, indice_faiss, trechos, modelo_embeddings,
                   quantidade_resultados=5):
    """
    Busca os trechos mais similares a uma consulta usando similaridade de cosseno.

    Argumentos:
        consulta (str): Texto da consulta (ex: "Amoxicilina com Ibuprofeno interação").
        indice_faiss (faiss.IndexFlatIP): Índice FAISS.
        trechos (list[dict]): Lista de trechos indexados.
        modelo_embeddings (SentenceTransformer): Modelo de embeddings.
        quantidade_resultados (int): Número de resultados a retornar.

    Retorna:
        list[dict]: Lista ordenada por similaridade decrescente. Cada item contém:
            - medicamento (str)
            - texto (str)
            - fonte (str)
            - nome_arquivo (str)
            - similaridade (float): Similaridade de cosseno (0 a 1)
    """
    # Gerar embedding da consulta
    embedding_consulta = modelo_embeddings.encode(
        [consulta],
        normalize_embeddings=True
    ).astype(np.float32)

    # Buscar no FAISS
    distancias, indices = indice_faiss.search(
        embedding_consulta,
        quantidade_resultados
    )

    resultados = []
    for distancia, indice in zip(distancias[0], indices[0]):
        trecho = trechos[indice]
        resultados.append({
            "medicamento": trecho["medicamento"],
            "texto": trecho["texto"],
            "fonte": trecho["fonte"],
            "nome_arquivo": trecho["nome_arquivo"],
            "similaridade": float(distancia),
        })

    return resultados
```

---

## 3. Busca Híbrida (BM25 + Embeddings)

### 3.1 Construção e combinação

```python
def construir_indice_bm25(trechos):
    """
    Constrói índice BM25 a partir dos textos dos trechos.

    O BM25 é um algoritmo de recuperação baseado em frequência de termos
    que pontua documentos pela ocorrência de palavras da consulta.
    """
    from rank_bm25 import BM25Okapi

    textos_tokenizados = [
        trecho["texto"].lower().split() for trecho in trechos
    ]
    return BM25Okapi(textos_tokenizados)


def buscar_trechos_hibrida(consulta, indice_faiss, indice_bm25, trechos,
                           modelo_embeddings, quantidade_resultados=5,
                           peso_cosseno=0.3):
    """
    Combina similaridade de cosseno (FAISS) com BM25 para busca híbrida.

    peso_cosseno=0.3 → 30% cosseno, 70% BM25.
    O BM25 tem peso maior porque nomes de medicamentos são termos exatos —
    se o usuário busca "AAS Protect", o BM25 encontra a correspondência
    literal, enquanto o embedding pode retornar trechos sobre
    "ácido acetilsalicílico" (sinônimo).
    """
    # Scores de cosseno (FAISS)
    embedding_consulta = modelo_embeddings.encode(
        [consulta], normalize_embeddings=True
    ).astype(np.float32)

    # Buscar todos para pontuar
    distancias, indices = indice_faiss.search(embedding_consulta, len(trechos))
    pontuacoes_cosseno = {}
    for distancia, indice in zip(distancias[0], indices[0]):
        pontuacoes_cosseno[indice] = float(distancia)

    # Scores BM25
    pontuacoes_bm25_brutas = indice_bm25.get_scores(consulta.lower().split())
    pontuacoes_bm25 = (
        (pontuacoes_bm25_brutas - pontuacoes_bm25_brutas.min())
        / (pontuacoes_bm25_brutas.max() - pontuacoes_bm25_brutas.min() + 1e-9)
    )

    # Combinar
    pontuacoes_combinadas = []
    for i in range(len(trechos)):
        cs = pontuacoes_cosseno.get(i, 0.0)
        bs = pontuacoes_bm25[i]
        combinada = peso_cosseno * cs + (1 - peso_cosseno) * bs
        pontuacoes_combinadas.append((i, combinada))

    pontuacoes_combinadas.sort(key=lambda x: x[1], reverse=True)

    resultados = []
    for indice, pontuacao in pontuacoes_combinadas[:quantidade_resultados]:
        trecho = trechos[indice]
        resultados.append({
            "medicamento": trecho["medicamento"],
            "texto": trecho["texto"],
            "fonte": trecho["fonte"],
            "nome_arquivo": trecho["nome_arquivo"],
            "similaridade_cosseno": pontuacoes_cosseno.get(indice, 0.0),
            "similaridade_bm25": float(pontuacoes_bm25[indice]),
            "similaridade_combinada": float(pontuacao),
        })

    return resultados
```

---

## 4. Comparação de Modelos de Embeddings

```python
def comparar_modelos_embeddings(trechos, consultas_teste, gabarito_relevancia):
    """
    Compara dois modelos de embeddings: BERT português vs MiniLM multilíngue.

    Métrica: Precisão@5 — quantos dos top-5 resultados são realmente relevantes.

    Argumentos:
        trechos: Lista de trechos.
        consultas_teste: Lista de consultas.
        gabarito_relevancia: Dict mapeando consulta → lista de índices relevantes.

    Retorna:
        dict: Métricas para cada modelo.
    """
    from sentence_transformers import SentenceTransformer

    modelos = {
        "bert_portugues": SentenceTransformer(
            "neuralmind/bert-base-portuguese-cased"
        ),
        "minilm_multilingue": SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    }

    resultados = {}

    for nome_modelo, modelo in modelos.items():
        if torch.cuda.is_available():
            modelo = modelo.to("cuda")

        registro.info("Avaliando modelo: %s", nome_modelo)
        embeddings = modelo.encode(
            [t["texto"] for t in trechos],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        indice = faiss.IndexFlatIP(embeddings.shape[1])
        indice.add(embeddings.astype(np.float32))

        precisoes = []
        for consulta, indices_relevantes in gabarito_relevancia.items():
            embedding_consulta = modelo.encode(
                [consulta], normalize_embeddings=True
            ).astype(np.float32)
            _, indices_encontrados = indice.search(embedding_consulta, 5)

            acertos = len(
                set(indices_encontrados[0]) & set(indices_relevantes)
            )
            precisoes.append(acertos / 5)

        resultados[nome_modelo] = {
            "precisao_media": sum(precisoes) / len(precisoes),
            "dimensao_embedding": embeddings.shape[1],
        }

    return resultados
```

---

## 5. Estrutura do Caderno (10 células)

| Célula | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | **Configuração:** logging (`caderno_03_embeddings.log`), imports, constantes |
| 3 | Code | `carregar_trechos_bulas()` — lê bulas, extrai seções, retorna lista de trechos. `registro.info("Trechos carregados: %d", len(trechos))` |
| 4 | Code + Markdown | **Embeddings + FAISS:** `criar_indice_busca()`, `buscar_trechos()`. Teste rápido: "Amoxicilina com Ibuprofeno". Log com scores. |
| 5 | Code + Markdown | **10 consultas de teste:** tabela com top-5 resultados por consulta. Análise qualitativa. |
| 6 | Code + Markdown | **Busca híbrida:** `buscar_trechos_hibrida()`. Comparação lado a lado com busca pura para 3 consultas. |
| 7 | Code + Markdown | **Comparação de 2 modelos:** BERT pt vs MiniLM. Tabela com Precisão@5, dimensão, tempo. |
| 8 | Markdown | **Análise de acertos e falhas:** 3 casos de acerto + 3 de falha com explicação |
| 9 | Markdown | **Justificativa da estratégia:** FAISS IndexFlatIP, top-k=5, peso_cosseno=0.3 |
| 10 | Code | `registro.info("Caderno 03 concluído. FAISS: %d vetores", indice_faiss.ntotal)` |

---

## 6. Commits Atômicos

### Commit 1: Carregamento de trechos
```
feat: Sprint 3 — carregar_trechos_bulas: leitura direta, extração de seções, chunking
```

### Commit 2: Embeddings + FAISS + busca semântica
```
feat: Sprint 3 — criar_indice_busca FAISS + buscar_trechos com cosseno
```

### Commit 3: Busca híbrida + 10 consultas de teste
```
feat: Sprint 3 — busca híbrida BM25+embeddings + 10 consultas com análise
```

### Commit 4: Comparação de modelos + análise de falhas + conclusão
```
feat: Sprint 3 — comparação BERT pt vs MiniLM, acertos/falhas, justificativa
```
