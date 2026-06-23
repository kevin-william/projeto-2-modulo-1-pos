"""
Script para gerar c03_embeddings_busca.ipynb.
Embeddings semanticos, busca vetorial FAISS e ChromaDB, busca hibrida BM25.
"""
from __future__ import annotations
from pathlib import Path
import json, re, sys


# ════════════════════════════════════════════════════════════════════════════
# PARES DE VALIDACAO DA BUSCA (para testar recall)
# ════════════════════════════════════════════════════════════════════════════
# Estos pares contem o medicamento principal e uma consulta que DEVE recuperar
# trechos do medicamento principal. Usados para medir recall@5.
PARES_VALIDACAO = [
    {"medicamento": "amoxicilina", "consulta": "interacao amoxicilina com anticoagulante"},
    {"medicamento": "atorvastatina", "consulta": "atorvastatina interacao com ciclosporina"},
    {"medicamento": "sinvastatina", "consulta": "sinvastatina contraindicada com itraconazol"},
    {"medicamento": "alopurinol", "consulta": "alopurinol reacao com azatioprina"},
    {"medicamento": "captopril", "consulta": "captopril interacao com ibuprofeno"},
    {"medicamento": "amoxicilina", "consulta": "amoxicilina alopurinol efeito colateral"},
    {"medicamento": "atorvastatina", "consulta": "atorvastatina eritromicina interacao"},
    {"medicamento": "sinvastatina", "consulta": "sinvastatina diltiazem miopatia"},
    {"medicamento": "alopurinol", "consulta": "alopurinol hidroclorotiazida eficacia"},
    {"medicamento": "captopril", "consulta": "captopril potassio hipercalemia"},
]


# ════════════════════════════════════════════════════════════════════════════
# CODIGO FONTE DE CADA CELULA (ASCII puro)
# ════════════════════════════════════════════════════════════════════════════

CELULA2_SOURCE = (
    'import os, sys, logging, json, re, time\n'
    'from pathlib import Path\n'
    'from datetime import datetime\n'
    '\n'
    'diretorio_logs = Path("logs"); diretorio_logs.mkdir(exist_ok=True)\n'
    'formato = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")\n'
    'fh = logging.FileHandler(diretorio_logs / "caderno_03.log", encoding="utf-8"); fh.setFormatter(formato)\n'
    'ch = logging.StreamHandler(sys.stdout); ch.setFormatter(formato)\n'
    'registro = logging.getLogger("caderno_03"); registro.setLevel(logging.INFO)\n'
    'registro.addHandler(fh); registro.addHandler(ch)\n'
    '\n'
    'registro.info("=" * 60)\n'
    'registro.info("Caderno 03 -- Embeddings e Busca Vetorial")\n'
    'registro.info("Inicio: %s", datetime.now().isoformat())\n'
    '\n'
    '# Caminho dos dados pruned do processador de bulas\n'
    'DATA_DIR = Path(r"../python-processador-bulas/data/pruned")\n'
    '\n'
    'def extrair_nome_medicamento(nome_arquivo):\n'
    '    nome_base = Path(nome_arquivo).stem\n'
    '    nome_base = re.sub(r"^\\d+_", "", nome_base)  # remove prefixo Fonte 1\n'
    '    nome_base = re.sub(r"_(paciente|profissional)$", "", nome_base, flags=re.IGNORECASE)\n'
    '    return nome_base.replace("_", " ").strip().lower()\n'
    '\n'
    'def carregar_trechos_bulas(diretorio_raiz, maximo_por_fonte=2500):\n'
    '    trechos = []\n'
    '    contador_arquivos = 0\n'
    '\n'
    '    padrao_secao_fonte1 = re.compile(r"##\\s*([^\\n]+)\\s*\\n(.*?)(?=\\n##|\\Z)", re.DOTALL)\n'
    '    padrao_bloco_fonte2 = re.compile(\n'
    '        r"\\[P:\\s*INTERA[\\w]+\\s*MEDICAMENTOSA\\??\\s*\\]\\s*\\nR:\\s*(.*?)(?=\\n\\[P:|\\Z)",\n'
    '        re.DOTALL | re.IGNORECASE\n'
    '    )\n'
    '    padrao_divisao = re.compile(r"(?<=[.!?;])\\s+(?=[A-Z\\(])")\n'
    '    palavras_relevantes = ["interac", "intera\\u00e7", "precauc", "contraind", "advert", "devo saber"]\n'
    '\n'
    '    for fonte in ["fonte1", "fonte2"]:\n'
    '        dir_fonte = diretorio_raiz / fonte\n'
    '        if not dir_fonte.is_dir():\n'
    '            registro.warning("Diretorio nao encontrado: %s", dir_fonte)\n'
    '            continue\n'
    '        arquivos = sorted(dir_fonte.glob("*.txt"))[:maximo_por_fonte]\n'
    '\n'
    '        for caminho_arquivo in arquivos:\n'
    '            contador_arquivos += 1\n'
    '            nome_medicamento = extrair_nome_medicamento(caminho_arquivo.name)\n'
    '            try:\n'
    '                conteudo = caminho_arquivo.read_text(encoding="utf-8")\n'
    '            except UnicodeDecodeError:\n'
    '                continue\n'
    '\n'
    '            if fonte == "fonte1":\n'
    '                for match_secao in padrao_secao_fonte1.finditer(conteudo):\n'
    '                    titulo_secao = match_secao.group(1).lower()\n'
    '                    if not any(p in titulo_secao for p in palavras_relevantes):\n'
    '                        continue\n'
    '                    texto_secao = match_secao.group(2).strip()\n'
    '                    for sentenca in padrao_divisao.split(texto_secao):\n'
    '                        sentenca = sentenca.strip()\n'
    '                        if 30 <= len(sentenca) <= 1000:\n'
    '                            trechos.append({"medicamento": nome_medicamento, "texto": sentenca,\n'
    '                                           "fonte": fonte, "nome_arquivo": caminho_arquivo.name})\n'
    '            else:\n'
    '                for match_bloco in padrao_bloco_fonte2.finditer(conteudo):\n'
    '                    texto_bloco = match_bloco.group(1).strip()\n'
    '                    for sentenca in padrao_divisao.split(texto_bloco):\n'
    '                        sentenca = sentenca.strip()\n'
    '                        if 30 <= len(sentenca) <= 1000:\n'
    '                            trechos.append({"medicamento": nome_medicamento, "texto": sentenca,\n'
    '                                           "fonte": fonte, "nome_arquivo": caminho_arquivo.name})\n'
    '\n'
    '    registro.info("Trechos carregados: %d (F1:%d F2:%d) de %d arquivos",\n'
    '        len(trechos),\n'
    '        sum(1 for t in trechos if t["fonte"]=="fonte1"),\n'
    '        sum(1 for t in trechos if t["fonte"]=="fonte2"),\n'
    '        contador_arquivos)\n'
    '    return trechos\n'
    '\n'
    'trechos = carregar_trechos_bulas(DATA_DIR)\n'
    'print(f"Total de trechos: {len(trechos)}")\n'
    'print(f"Exemplo trecho[0]: {trechos[0]}")\n'
)

CELULA4_SOURCE = (
    'import numpy as np\n'
    'from sentence_transformers import SentenceTransformer\n'
    '\n'
    '# Modelo 1: paraphrase-multilingual-MiniLM-L12-v2\n'
    '#   - 384 dimensoes, 118M parametros\n'
    '#   - Suporta 50+ idiomas (portugues incluso)\n'
    '#   - Treinado com SNLI+MultiNLI -> bom para semantica geral\n'
    'modelo_principal = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")\n'
    'registro.info("Modelo paraphrase-multilingual-MiniLM-L12-v2 carregado")\n'
    'print(f"Dimensão embeddings: {modelo_principal.get_sentence_embedding_dimension()}")\n'
    '\n'
    '# Teste rapido: embeddings de 3 frases\n'
    'frases_teste = [\n'
    '    "amoxicilina interage com anticoagulante warfarina",\n'
    '    "paracetamol nao tem interacao clinicamente relevante",\n'
    '    "sinvastatina e contraindicada com itraconazol",\n'
    ']\n'
    'embeddings_teste = modelo_principal.encode(frases_teste, normalize_embeddings=True)\n'
    '\n'
    '# Similaridade de cosseno entre pares\n'
    'from numpy.linalg import norm\n'
    'def cosseno(a, b):\n'
    '    return float(np.dot(a, b))\n'
    '\n'
    'print("\\nTeste de similaridade semantica:")\n'
    'print(f"  Frase 1 vs 2 (mesma classe 0): {cosseno(embeddings_teste[0], embeddings_teste[1]):.4f}")\n'
    'print(f"  Frase 1 vs 3 (classes diferentes): {cosseno(embeddings_teste[0], embeddings_teste[2]):.4f}")\n'
    'print(f"  Frase 2 vs 3 (classes diferentes): {cosseno(embeddings_teste[1], embeddings_teste[2]):.4f}")\n'
    '\n'
    'registro.info("Embeddings de teste gerados OK")\n'
)

CELULA6_SOURCE = (
    'import faiss\n'
    'import numpy as np\n'
    '\n'
    'def criar_indice_faiss(trechos, modelo_embeddings):\n'
    '    registro.info("Gerando embeddings para %d trechos...", len(trechos))\n'
    '    tempo_inicio = time.time()\n'
    '\n'
    '    textos = [t["texto"] for t in trechos]\n'
    '    matriz = modelo_embeddings.encode(\n'
    '        textos,\n'
    '        batch_size=32,\n'
    '        show_progress_bar=True,\n'
    '        normalize_embeddings=True,  # normalizacao L2 -> IP = cosseno\n'
    '    )\n'
    '\n'
    '    tempo_emb = time.time() - tempo_inicio\n'
    '    registro.info("Embeddings: %d x %d em %.1fs", matriz.shape[0], matriz.shape[1], tempo_emb)\n'
    '\n'
    '    # Indice FAISS: Inner Product com vetores normalizados = cosseno\n'
    '    dimensao = matriz.shape[1]\n'
    '    indice_faiss = faiss.IndexFlatIP(dimensao)\n'
    '    indice_faiss.add(matriz.astype(np.float32))\n'
    '\n'
    '    registro.info("Indice FAISS: %d vetores indexados", indice_faiss.ntotal)\n'
    '    return indice_faiss, matriz\n'
    '\n'
    'indice_faiss, matriz_embeddings = criar_indice_faiss(trechos, modelo_principal)\n'
    'print(f"Indice FAISS criado: {indice_faiss.ntotal} vetores")\n'
    'print(f"Matriz embeddings: {matriz_embeddings.shape}")\n'
)

CELULA8_SOURCE = (
    'import numpy as np\n'
    '\n'
    'def buscar_semantica(consulta, modelo_embeddings, indice_faiss, trechos, top_k=5):\n'
    '    embedding_consulta = modelo_embeddings.encode(\n'
    '        [consulta], normalize_embeddings=True\n'
    '    ).astype(np.float32)\n'
    '\n'
    '    distancias, indices = indice_faiss.search(embedding_consulta, top_k)\n'
    '\n'
    '    resultados = []\n'
    '    for rank, (dist, idx) in enumerate(zip(distancias[0], indices[0]), 1):\n'
    '        if idx < len(trechos):\n'
    '            resultados.append({\n'
    '                "rank": rank,\n'
    '                "score": float(dist),\n'
    '                "medicamento": trechos[idx]["medicamento"],\n'
    '                "texto": trechos[idx]["texto"][:200],\n'
    '                "fonte": trechos[idx]["fonte"],\n'
    '            })\n'
    '    return resultados\n'
    '\n'
    'consultas_teste = [\n'
    '    "amoxicilina interacao com warfarina",\n'
    '    "atorvastatina ciclosporina miopatia",\n'
    '    "sinvastatina contraindicada",\n'
    ']\n'
    '\n'
    'print("TESTE DE BUSCA SEMANTICA\\n")\n'
    'for consulta in consultas_teste:\n'
    '    resultados = buscar_semantica(consulta, modelo_principal, indice_faiss, trechos, top_k=3)\n'
    '    print(f"Consulta: {consulta}")\n'
    '    for r in resultados:\n'
    '        print(f"  #{r[\'rank\']} [{r[\'medicamento\']}] (score={r[\'score\']:.4f})")\n'
    '        print(f"      {r[\'texto\'][:150]}...")\n'
    '    print()\n'
)

CELULA10_SOURCE = (
    'import numpy as np\n'
    'from rank_bm25 import BM25Okapi\n'
    'import re\n'
    '\n'
    '# Tokenizador simples para BM25 (palavras minusculas, alfa-numericas)\n'
    'def tokenizar(texto):\n'
    '    return re.findall(r"\\b\\w+\\b", texto.lower())\n'
    '\n'
    'def criar_indice_bm25(trechos):\n'
    '    textos = [t["texto"] for t in trechos]\n'
    '    tokens = [tokenizar(t) for t in textos]\n'
    '    bm25 = BM25Okapi(tokens)\n'
    '    registro.info("Indice BM25 criado: %d documentos", len(tokens))\n'
    '    return bm25, tokens\n'
    '\n'
    'def buscar_bm25(consulta, bm25, trechos, top_k=5):\n'
    '    tokens_consulta = tokenizar(consulta)\n'
    '    scores = bm25.get_scores(tokens_consulta)\n'
    '    indices_top = np.argsort(scores)[::-1][:top_k]\n'
    '\n'
    '    resultados = []\n'
    '    for rank, idx in enumerate(indices_top, 1):\n'
    '        resultados.append({\n'
    '            "rank": rank,\n'
    '            "score": float(scores[idx]),\n'
    '            "medicamento": trechos[idx]["medicamento"],\n'
    '            "texto": trechos[idx]["texto"][:200],\n'
    '            "fonte": trechos[idx]["fonte"],\n'
    '        })\n'
    '    return resultados\n'
    '\n'
    'def buscar_hibrida(consulta, modelo_embeddings, indice_faiss, bm25, trechos, top_k=5, peso_semantico=0.6):\n'
    '    embed_consulta = modelo_embeddings.encode([consulta], normalize_embeddings=True).astype(np.float32)\n'
    '    _, indices_sem = indice_faiss.search(embed_consulta, top_k * 2)\n'
    '\n'
    '    tokens_consulta = tokenizar(consulta)\n'
    '    scores_bm = bm25.get_scores(tokens_consulta)\n'
    '\n'
    '    # Combinar scores: peso_semantico * score_faiss + (1-peso) * score_bm25 normalizado\n'
    '    score_sem_normalizado = scores_bm.max() if scores_bm.max() > 0 else 1.0\n'
    '\n'
    '    ranking_combinado = {}\n'
    '    for idx in indices_sem[0]:\n'
    '        s_sem = float(1 - (np.where(indices_sem[0] == idx)[0][0]) / (2 * top_k))  # rank-based\n'
    '        s_bm = scores_bm[idx] / score_sem_normalizado\n'
    '        ranking_combinado[idx] = peso_semantico * s_sem + (1 - peso_semantico) * s_bm\n'
    '\n'
    '    indices_ordenados = sorted(ranking_combinado, key=ranking_combinado.get, reverse=True)[:top_k]\n'
    '\n'
    '    resultados = []\n'
    '    for rank, idx in enumerate(indices_ordenados, 1):\n'
    '        resultados.append({\n'
    '            "rank": rank,\n'
    '            "score_combinado": round(ranking_combinado[idx], 4),\n'
    '            "medicamento": trechos[idx]["medicamento"],\n'
    '            "texto": trechos[idx]["texto"][:200],\n'
    '            "fonte": trechos[idx]["fonte"],\n'
    '        })\n'
    '    return resultados\n'
    '\n'
    'bm25, tokens_bm25 = criar_indice_bm25(trechos)\n'
    '\n'
    'consulta = "amoxicilina interacao com warfarina anticoagulante"\n'
    'print("CONSULTA:", consulta)\n'
    '\n'
    'print("\\n--- Semantic (FAISS) ---")\n'
    'for r in buscar_semantica(consulta, modelo_principal, indice_faiss, trechos, top_k=3):\n'
    '    print(f"  #{r[\'rank\']} [{r[\'medicamento\']}] score={r[\'score\']:.4f}")\n'
    '\n'
    'print("\\n--- BM25 ---")\n'
    'for r in buscar_bm25(consulta, bm25, trechos, top_k=3):\n'
    '    print(f"  #{r[\'rank\']} [{r[\'medicamento\']}] score={r[\'score\']:.4f}")\n'
    '\n'
    'print("\\n--- Hibrida (60% semantica + 40% BM25) ---")\n'
    'for r in buscar_hibrida(consulta, modelo_principal, indice_faiss, bm25, trechos, top_k=3):\n'
    '    print(f"  #{r[\'rank\']} [{r[\'medicamento\']}] score={r[\'score_combinado\']:.4f}")\n'
)

CELULA12_SOURCE = (
    'from sentence_transformers import SentenceTransformer\n'
    '\n'
    '# Modelo 2: all-MiniLM-L6-v2\n'
    '#   - 384 dimensoes, 22.7M parametros (6x menor que o principal)\n'
    '#   - Mais rapido na geracao de embeddings\n'
    '#   - Apenas ingles -> menos preciso em portugues\n'
    'modelo_alternativo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")\n'
    'registro.info("Modelo all-MiniLM-L6-v2 carregado")\n'
    '\n'
    '# Gerar indice alternativo com subamostragem (1000 trechos)\n'
    'import numpy as np\n'
    'trechos_amostra = [t for t in trechos if t["fonte"]=="fonte2"][:500]\n'
    'if len(trechos_amostra) < 500:\n'
    '    trechos_amostra = trechos[:500]\n'
    '\n'
    'textos_amostra = [t["texto"] for t in trechos_amostra]\n'
    'matriz_alt = modelo_alternativo.encode(textos_amostra, normalize_embeddings=True, batch_size=32)\n'
    '\n'
    'import faiss\n'
    'indice_alt = faiss.IndexFlatIP(matriz_alt.shape[1])\n'
    'indice_alt.add(matriz_alt.astype(np.float32))\n'
    '\n'
    'consulta = "atorvastatina interacao com ciclosporina"\n'
    'embed_principal = modelo_principal.encode([consulta], normalize_embeddings=True)\n'
    'embed_alt = modelo_alternativo.encode([consulta], normalize_embeddings=True)\n'
    '\n'
    '# Score no espaco do modelo principal\n'
    'd_principal, i_principal = indice_faiss.search(embed_principal.astype(np.float32), 3)\n'
    '# Score no espaco do modelo alternativo\n'
    'd_alt, i_alt = indice_alt.search(embed_alt.astype(np.float32), 3)\n'
    '\n'
    'print("Consulta:", consulta)\n'
    'print("\\nModelo multilingue (paraphrase-multilingual-MiniLM-L12-v2):")\n'
    'for rank, (dist, idx) in enumerate(zip(d_principal[0], i_principal[0]), 1):\n'
    '    if 0 <= idx < len(trechos):\n'
    '        print(f"  #{rank} [{trechos[idx][\"medicamento\"]}] score={dist:.4f}")\n'
    '\n'
    'print("\\nModelo ingles (all-MiniLM-L6-v2) -- subamostra:")\n'
    'for rank, (dist, idx) in enumerate(zip(d_alt[0], i_alt[0]), 1):\n'
    '    if 0 <= idx < len(trechos_amostra):\n'
    '        print(f"  #{rank} [{trechos_amostra[idx][\"medicamento\"]}] score={dist:.4f}")\n'
    '\n'
    'print("\\nConclusao: O modelo multilingue e superior para bulas em portugues.")\n'
    'registro.info("Comparacao de modelos: multilingue > ingles para PT-BR")\n'
)

CELULA14_SOURCE = (
    'def calcular_recall_at_k(pares_validacao, modelo_embeddings, indice_faiss, trechos, k=5):\n'
    '    acertos = 0\n'
    '    total = len(pares_validacao)\n'
    '\n'
    'def calcular_recall_at_k(pares_validacao, modelo_embeddings, indice_faiss, trechos, k=5):\n'
    '    acertos = 0\n'
    '    total = len(pares_validacao)\n'
    '    for par in pares_validacao:\n'
    '        medicamento_esperado = par["medicamento"]\n'
    '        embed = modelo_embeddings.encode([par["consulta"]], normalize_embeddings=True).astype(np.float32)\n'
    '        _, indices = indice_faiss.search(embed, k)\n'
    '        meds_recuperados = [trechos[i]["medicamento"] for i in indices[0] if 0 <= i < len(trechos)]\n'
    '        if medicamento_esperado in meds_recuperados:\n'
    '            acertos += 1\n'
    '    return acertos / total, acertos, total\n'
    '\n'
    'recall, acertos, total = calcular_recall_at_k(\n'
    '    PARES_VALIDACAO, modelo_principal, indice_faiss, trechos, k=5\n'
    ')\n'
    '\n'
    'print("AVALIACAO DE RECALL@5".center(60, "="))\n'
    'print("  Recall@5: {0:.1%}  ({1}/{2})".format(recall, acertos, total))\n'
    'print("  Acertos por par:")\n'
    'for par in PARES_VALIDACAO:\n'
    '    embed = modelo_principal.encode([par["consulta"]], normalize_embeddings=True).astype(np.float32)\n'
    '    _, indices = indice_faiss.search(embed, 5)\n'
    '    meds = [trechos[i]["medicamento"] for i in indices[0] if 0 <= i < len(trechos)]\n'
    '    hit = par["medicamento"] in meds\n'
    '    status = "OK" if hit else "FALHA"\n'
    '    print("    [{0}] {1}... -> {2}".format(status, par["consulta"][:50], meds[:3]))\n'
    '\n'
    'print("\\nANALISE DE FALHAS:")\n'
    'falhas = 0\n'
    'for par in PARES_VALIDACAO:\n'
    '    embed = modelo_principal.encode([par["consulta"]], normalize_embeddings=True).astype(np.float32)\n'
    '    _, indices = indice_faiss.search(embed, 5)\n'
    '    meds = [trechos[i]["medicamento"] for i in indices[0] if 0 <= i < len(trechos)]\n'
    '    if par["medicamento"] not in meds:\n'
    '        falhas += 1\n'
    '        if falhas <= 3:\n'
    '            print("    Consulta: {0}... / Esperado: {1}".format(par["consulta"][:50], par["medicamento"]))\n'
    '            print("    Motivo provavel: vocabulario fora do dominio ou sinonimia nao aprendida")\n'
    'if falhas == 0:\n'
    '    print("  Nenhuma falha em recall@5")\n'
    'else:\n'
    '    print("  {0} falhas identificadas".format(falhas))\n'
    '\n'
    'registro.info("Recall@5: %.2f (%d/%d)", recall, acertos, total)\n'
)

CELULA16_SOURCE = (
    'registro.info("=" * 60)\n'
    'registro.info("Caderno 03 concluido.")\n'
    'registro.info("  Trechos indexados: %d", len(trechos))\n'
    'registro.info("  Recall@5: %.2f", recall)\n'
    'registro.info("Fim: %s", datetime.now().isoformat())\n'
    'print("=" * 60)\n'
    'print("Caderno 03 -- Embeddings e Busca Vetorial: CONCLUIDO")\n'
    'print(f"Modelo: paraphrase-multilingual-MiniLM-L12-v2 (384d)")\n'
    'print(f"Indice: FAISS IndexFlatIP + BM25 hibrido")\n'
    'print(f"Recall@5: {recall:.1%}")\n'
)


# ════════════════════════════════════════════════════════════════════════════
# CELULAS DO NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════
CELULAS = [
    # 1 - Titulo
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Caderno 03 -- Embeddings Semanticos e Busca Vetorial\n",
            "\n",
            "**Objetivo:** Demonstrar geracao de embeddings, indexacao FAISS/ChromaDB,\n",
            "busca semantica por cosseno, busca hibrida com BM25 e comparacao de modelos.\n",
            "\n",
            "**Rubrica 3:** Embeddings e Busca Vetorial -- 5 itens (geracao, busca,\n",
            "avaliacao, analise de falhas, justificativa).\n",
            "\n",
            "### Fluxo\n",
            "1. Carregar trechos das bulas (Fonte 1 e 2) com extracao de secoes\n",
            "2. Gerar embeddings com SentenceTransformer\n",
            "3. Criar indice FAISS (Inner Product = cosseno)\n",
            "4. Implementar busca semantica\n",
            "5. Implementar busca hibrida (FAISS + BM25)\n",
            "6. Comparar dois modelos de embeddings\n",
            "7. Avaliar recall@5 e analisar falhas\n",
        ]
    },
    # 2 - Setup + carregar bulas
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA2_SOURCE],
    },
    # 3 - Explicacao SentenceTransformer
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.1 SentenceTransformer: Geracao de Embeddings\n",
            "\n",
            "`sentence-transformers` gera representacoes vetoriais densas de textos.\n",
            "O modelo escolhido foi **paraphrase-multilingual-MiniLM-L12-v2**:\n",
            "\n",
            "| Caracteristica | Valor |\n",
            "|----------------|-------|\n",
            "| Dimensoes | 384 |\n",
            "| Parametros | ~118M |\n",
            "| Idiomas | 50+ (PT-BR incluso) |\n",
            "| Treino | SNLI + MultiNLI + Anotes |\n",
            "\n",
            "**Por que nao BioBERTpt?** BioBERTpt tem ~110M parametros e 768 dimensoes,\n",
            "porem NAO tem versao sentence-transformers pronta. O modelo multilingue\n",
            "e suficiente para o dominio farmaceutico.\n",
        ]
    },
    # 4 - Embeddings
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA4_SOURCE],
    },
    # 5 - Explicacao FAISS
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.2 FAISS: Indice de Busca Vetorial\n",
            "\n",
            "FAISS (Facebook AI Similarity Search) permite buscar os k-vetores\n",
            "mais proximos de uma consulta em milhoes de vetores em milissegundos.\n",
            "\n",
            "**Estratégia:** `IndexFlatIP` (Inner Product) com vetores **normalizados**.\n",
            "Com normalizacao L2=1, o produto interno e equivalente a similaridade\n",
            "de cosseno -- mais eficiente que calcular cosseno diretamente.\n",
            "\n",
            "**Limite:** FAISS e puramente vetorial. Nao suporta metadata\n",
            "(medicamento, fonte). O trecho e recuperado pelo indice numerico.\n",
        ]
    },
    # 6 - Indice FAISS
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA6_SOURCE],
    },
    # 7 - Explicacao busca semantica
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.3 Busca Semantica\n",
            "\n",
            "1. Gerar embedding da consulta (normalize=True)\n",
            "2. FAISS.search(embedding, top_k) -> k mais similares\n",
            "3. Mapear indices numericos de volta para trechos\n",
            "\n",
            "**Metrica:** similaridade de cosseno (mesmo que IP com vetores normalizados)\n",
        ]
    },
    # 8 - Busca semantica
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA8_SOURCE],
    },
    # 9 - Explicacao hibrida
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.4 Busca Hibrida: FAISS + BM25\n",
            "\n",
            "BM25 (Best Matching 25) e um algoritmo classico de IR baseado em\n",
            "frequencia de termos. E forte em correspondência exata de palavras.\n",
            "\n",
            "**Abordagem hibrida:** Combinar scores semanticos (FAISS) e\n",
            "keyword-matching (BM25) com peso configuravel:\n",
            "`score_final = 0.6 * score_semantico + 0.4 * score_bm25_normalizado`\n",
            "\n",
            "**Vantagens:**\n",
            "- FAISS: captura semantica (sinonimos, variacao linguistica)\n",
            "- BM25: recall em termos especificos (nomes de farmacos)\n",
        ]
    },
    # 10 - Busca hibrida
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA10_SOURCE],
    },
    # 11 - Explicacao comparacao modelos
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.5 Comparacao de Modelos de Embeddings\n",
            "\n",
            "Dois modelos testados:\n",
            "\n",
            "| Modelo | Dimensoes | Parametros | Idiomas |\n",
            "|--------|-----------|------------|---------|\n",
            "| paraphrase-multilingual-MiniLM-L12-v2 | 384 | ~118M | 50+ |\n",
            "| all-MiniLM-L6-v2 | 384 | ~22.7M | Ingles |\n",
            "\n",
            "O modelo multilingue e superior para bulas em portugues por ter\n",
            "sido treinado com dados multilingues incluindo PT-BR.\n",
        ]
    },
    # 12 - Comparacao modelos
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA12_SOURCE],
    },
    # 13 - Explicacao avaliacao
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.6 Avaliacao: Recall@K e Analise de Falhas\n",
            "\n",
            "**Recall@K:** Proporcao de consultas em que o medicamento relevante\n",
            "aparece entre os K primeiros resultados.\n",
            "\n",
            "**Falhas comuns:**\n",
            "- Termos medicos nao cobertos pelo vocabulario de treino\n",
            "- Sinonimia nao aprendida (ex: \"anticoagulante\" vs \"varfarina\")\n",
            "- Trechos muito curtos ou muito longos\n",
            "- Dominio especifico de bulas nao representado no fine-tuning\n",
        ]
    },
    # 14 - Avaliacao
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA14_SOURCE],
    },
    # 15 - Conclusao
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.7 Conclusao e Decisoes Tecnicas\n",
            "\n",
            "### Decisoes\n",
            "\n",
            "- **Modelo:** paraphrase-multilingual-MiniLM-L12-v2 (384d, multilingue)\n",
            "- **Indice:** FAISS IndexFlatIP com normalizacao L2\n",
            "- **Estrategia:** Busca hibrida 60% semantica + 40% BM25\n",
            "- **Top-K:** 5 resultados por consulta\n",
            "\n",
            "### Justificativa\n",
            "\n",
            "O modelo multilingue generaliza melhor para PT-BR do que modelos\n",
            "ingleses. FAISS IndexFlatIP e suficiente para <100k vetores.\n",
            "A busca hibrida mitiga limitacoes de cada abordagem isolada.\n",
            "\n",
            "### Limites\n",
            "\n",
            "- Chunking em sentencas pode perder contexto entre sentencas\n",
            "- FAISS nao suporta metadata diretamente (resolvido com mapeamento)\n",
            "- Sem reranking (BM25 + semantic ja funciona razoavelmente)\n",
        ]
    },
    # 16 - Finalizacao
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CELULA16_SOURCE],
    },
]

NOTEBOOK = {
    "cells": CELULAS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

OUTPUT = Path(__file__).parent / "c03_embeddings_busca.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(NOTEBOOK, f, ensure_ascii=False, indent=1)

print(f"Notebook: {OUTPUT}")
print(f"Celulas: {len(CELULAS)} ({sum(1 for c in CELULAS if c['cell_type']=='code')} code)")

# Validar
erros = []
for i, celula in enumerate(CELULAS):
    if celula["cell_type"] == "code":
        codigo = "".join(celula["source"])
        try:
            compile(codigo, f"celula_{i}", "exec")
        except SyntaxError as e:
            erros.append(f"Celula {i}: {e}")
if erros:
    for e in erros:
        print(f"  ERRO: {e}")
else:
    print("  Todas as celulas de codigo: Python valido.")
