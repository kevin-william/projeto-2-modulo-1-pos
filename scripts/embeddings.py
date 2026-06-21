"""Embedding e busca vetorial para o pipeline RAG.

Gera embeddings dos chunks de bulas usando SentenceTransformer,
indexa no ChromaDB e implementa busca hibrida (cosseno + BM25).

Fase 6 — Notebook 03: Embeddings + ChromaDB.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.config import CHUNKS_BULAS, ANOTACOES_DIR

log = logging.getLogger(__name__)

# ─── Configuração de Embeddings ────────────────────────────────────────

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "bulas_interacoes"

# Modelos SentenceTransformer testados
MODELS = {
    "BERTpt": "sentence-transformers/distilbert-multilingual-nli-stsb",
    "E5": "intfloat/e5-base-v2",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
}

# ─── Normalização de Texto ─────────────────────────────────────────────

def normalizar(texto: str) -> str:
    """Normaliza texto para busca."""
    if not texto:
        return ""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ─── Carregamento de Chunks ─────────────────────────────────────────────

def carregar_chunks(caminho: str | Path = None) -> pd.DataFrame:
    """Carrega os chunks de bulas do arquivo JSONL.

    Returns:
        DataFrame com colunas: id, texto, medicamento, fonte
    """
    caminho = Path(caminho) if caminho else CHUNKS_BULAS
    registros = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            obj = json.loads(linha)
            # Compatibilidade com ambos formatos de Fase 1
            chunk_id = obj.get("chunk_id") or hashlib.md5(
                obj.get("texto", "").encode()).hexdigest()[:8]
            registros.append({
                "id": chunk_id,
                "texto": obj.get("texto", ""),
                "medicamento": obj.get("medicamento", ""),
                "fonte": obj.get("fonte", "fonte1"),
            })
    df = pd.DataFrame(registros)
    log.info("Carregados %d chunks de %s", len(df), caminho)
    return df


# ─── Embeddings ─────────────────────────────────────────────────────────

def gerar_embeddings(
    textos: list[str],
    modelo_nome: str = "BERTpt",
    batch_size: int = 64,
) -> np.ndarray:
    """Gera embeddings usando SentenceTransformer.

    Args:
        textos: lista de textos para embeddar
        modelo_nome: chave em MODELS
        batch_size: tamanho do batch

    Returns:
        array numpy (N x dim)
    """
    model_key = MODELS.get(model_nome, model_nome)
    cache_folder = Path("data/modelos_cache")
    cache_folder.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_key, cache_folder=str(cache_folder))

    log.info("Gerando embeddings com %s (%dd)", model_nome, model.get_sentence_embedding_dimension())
    t0 = time.time()
    embeddings = model.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    log.info("Embeddings gerados em %.1fs — shape: %s", time.time() - t0, embeddings.shape)
    return embeddings


# ─── ChromaDB ─────────────────────────────────────────────────────────

def criar_collection(
    collection_name: str = COLLECTION_NAME,
    persist_directory: str | Path = CHROMA_DIR,
) -> Any:
    """Cria ou abre collection no ChromaDB."""
    import chromadb
    from chromadb.config import Settings

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"descricao": "Chunks de bulas para busca de interacoes medicamentosas"},
    )
    log.info("Collection '%s' pronta — count: %s", collection_name, collection.count())
    return collection


def indexar_chunks(
    collection,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> None:
    """Indexa chunks + embeddings no ChromaDB."""
    total = len(df)
    for i in range(0, total, batch_size):
        batch = df.iloc[i : i + batch_size]
        emb_batch = embeddings[i : i + batch_size]

        ids = [str(x) for x in batch["id"].tolist()]
        documentos = batch["texto"].tolist()
        metadados = [
            {"medicamento": str(r["medicamento"]), "fonte": str(r["fonte"])}
            for _, r in batch.iterrows()
        ]

        collection.upsert(ids=ids, documents=documentos, metadatas=metadados, embeddings=emb_batch.tolist())

    log.info("Indexados %d chunks no ChromaDB (total: %s)", total, collection.count())


def buscar_chunks(
    collection,
    query_embedding: np.ndarray,
    n: int = 5,
    filtro_medicamento: str | None = None,
) -> list[dict]:
    """Busca top-n chunks por similaridade de cosseno.

    Args:
        collection: ChromaDB collection
        query_embedding: embedding da consulta (1D)
        n: numero de resultados
        filtro_medicamento: filtra por nome do medicamento (opcional)

    Returns:
        Lista de dicts com id, texto, distancia, medicamento, fonte
    """
    where = {"medicamento": filtro_medicamento} if filtro_medicamento else None

    resultados = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = resultados.get("documents", [[]])[0]
    metas = resultados.get("metadatas", [{}])[0]
    dists = resultados.get("distances", [[]])[0]

    return [
        {
            "id": resultados["ids"][0][i],
            "texto": docs[i] if i < len(docs) else "",
            "distancia": dists[i] if i < len(dists) else 0.0,
            "medicamento": metas[i].get("medicamento", "") if i < len(metas) else "",
            "fonte": metas[i].get("fonte", "") if i < len(metas) else "",
        }
        for i in range(len(docs))
    ]


# ─── BM25 ──────────────────────────────────────────────────────────────

def bm25_score(consulta: str, documento: str, k1: float = 1.5, b: float = 0.75) -> float:
    """Calcula score BM25 entre consulta e documento.

    Implementação simplificada para comparação híbrida.
    """
    consulta_tokens = normalizar(consulta).split()
    doc_tokens = normalizar(documento).split()
    if not consulta_tokens or not doc_tokens:
        return 0.0

    # IDF simples: log((N - n + 0.5) / (n + 0.5))
    doc_set = set(doc_tokens)
    N = 1  # para um único documento
    idf_sum = 0.0
    for t in consulta_tokens:
        n = 1 if t in doc_set else 0
        idf = max(np.log(((N - n + 0.5) / (n + 0.5)) + 1), 0)
        tf = doc_tokens.count(t)
        denom = tf + k1 * (1 - b + b * len(doc_tokens) / 5)
        idf_sum += idf * (tf * (k1 + 1)) / denom
    return idf_sum


def busca_hibrida(
    collection,
    consulta: str,
    query_embedding: np.ndarray,
    n: int = 5,
    alpha: float = 0.3,
    filtro_medicamento: str | None = None,
) -> list[dict]:
    """Busca híbrida combinando cosseno e BM25.

    Score final = alpha * cos_score + (1-alpha) * bm25_normalizado

    Args:
        consulta: texto da consulta
        query_embedding: embedding da consulta
        alpha: peso do cosseno (0=só BM25, 1=só vetorial)
        n: número de resultados
    """
    # Busca vetorial
    vetoriais = buscar_chunks(collection, query_embedding, n * 2, filtro_medicamento)

    if not vetoriais:
        return []

    # Score BM25 para cada resultado
    bm25_scores = []
    for r in vetoriais:
        score = bm25_score(consulta, r["texto"])
        bm25_scores.append(score)

    max_bm25 = max(bm25_scores) if bm25_scores else 1.0

    # Combinação linear
    resultados_finais = []
    for r, bm25 in zip(vetoriais, bm25_scores):
        cos_score = 1.0 - r["distancia"]  # ChromaDB usa L2
        bm25_norm = bm25 / max_bm25 if max_bm25 > 0 else 0.0
        score_final = alpha * cos_score + (1 - alpha) * bm25_norm
        r["score"] = round(score_final, 4)
        r["cos_score"] = round(cos_score, 4)
        r["bm25_score"] = round(bm25_norm, 4)
        resultados_finais.append(r)

    resultados_finais.sort(key=lambda x: x["score"], reverse=True)
    return resultados_finais[:n]


# ─── Avaliação de Modelos ───────────────────────────────────────────────

def avaliar_modelos(
    collection,
    consultas: list[tuple[str, list[str]]],
    modelos: list[str] = None,
) -> pd.DataFrame:
    """Avalia múltiplos modelos de embedding.

    Args:
        collection: ChromaDB collection
        consultas: lista de (consulta, lista de_ids_esperados)
        modelos: lista de nomes de modelos (chaves de MODELS)

    Returns:
        DataFrame com Precision@3, MRR e latência por modelo
    """
    from sentence_transformers import SentenceTransformer

    modelos = modelos or list(MODELS.keys())
    cache_folder = Path("data/modelos_cache")

    resultados = []
    for nome in modelos:
        model_key = MODELS[nome]
        model = SentenceTransformer(model_key, cache_folder=str(cache_folder))

        precisions = []
        mrrs = []
        latencias = []

        for consulta, ids_esperados in consultas:
            t0 = time.time()
            emb = model.encode([consulta], normalize_embeddings=True)[0]
            lat = (time.time() - t0) * 1000

            top5 = buscar_chunks(collection, emb, n=5)
            ids_retornados = [r["id"] for r in top5]

            # P@3
            p3 = len(set(ids_retornados[:3]) & set(ids_esperados[:3])) / 3
            precisions.append(p3)

            # MRR
            mrr = 0.0
            for rank, rid in enumerate(ids_retornados, 1):
                if rid in ids_esperados:
                    mrr = 1.0 / rank
                    break
            mrrs.append(mrr)
            latencias.append(lat)

        resultados.append({
            "modelo": nome,
            "dimensoes": model.get_sentence_embedding_dimension(),
            "precision_at_3": round(np.mean(precisions), 3),
            "mrr": round(np.mean(mrrs), 3),
            "latencia_ms": round(np.mean(latencias), 0),
            "n_consultas": len(consultas),
        })
        log.info("Modelo %s — P@3: %.3f | MRR: %.3f | Lat: %.0fms", nome, resultados[-1]["precision_at_3"], resultados[-1]["mrr"], resultados[-1]["latencia_ms"])

    return pd.DataFrame(resultados)


# ─── Pipeline Principal ─────────────────────────────────────────────────

def construir_index(
    chunks_path: str | Path = None,
    modelo_embedding: str = "MiniLM",
    recreate: bool = False,
) -> Any:
    """Constrói (ou recarrega) o índice ChromaDB.

    Args:
        chunks_path: caminho do JSONL de chunks
        modelo_embedding: qual modelo usar para embeddings
        recreate: se True, recria o índice do zero

    Returns:
        ChromaDB collection
    """
    import chromadb
    from chromadb.config import Settings

    collection_name = COLLECTION_NAME
    persist_directory = CHROMA_DIR

    if recreate and persist_directory.exists():
        import shutil
        shutil.rmtree(persist_directory)
        log.info("Índice ChromaDB apagado para reconstrução.")

    # Tentar reabrir se já existe
    if not recreate:
        try:
            client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection(name=collection_name)
            if collection.count() > 0:
                log.info("Índice ChromaDB existente encontrado (%d chunks).", collection.count())
                return collection
        except Exception:
            pass

    # Construir do zero
    df = carregar_chunks(chunks_path)
    textos = df["texto"].tolist()
    embeddings = gerar_embeddings(textos, modelo_nome=modelo_embedding)
    collection = criar_collection(collection_name, persist_directory)
    indexar_chunks(collection, df, embeddings)
    log.info("Índice ChromaDB construído com sucesso.")
    return collection


# ─── Demonstração de Buscas ─────────────────────────────────────────────

def demonstrar_buscas(collection, consultas_teste: list[str] = None) -> None:
    """Demonstra buscas com análise de acertos e falhas."""
    consultas_teste = consultas_teste or [
        "sinvastatina itraconazol interacao",
        "amoxicilina penicilina alergia",
        "metformina insuficiencia renal",
        "warfarina sangramento",
        "ibuprofeno paracetamol dor",
        "losartana potasio",
        "omeprazol cisaprida",
        "fluoxetina tramadol",
        "atorvastatina gemfibrozil",
        "dexametasona anticoncepcional",
    ]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELS["MiniLM"], cache_folder="data/modelos_cache")

    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO DE BUSCAS VETORIAIS")
    print("=" * 70)

    for consulta in consultas_teste:
        emb = model.encode([consulta], normalize_embeddings=True)[0]
        top5 = busca_hibrida(collection, consulta, emb, n=5, alpha=0.3)

        print(f"\nConsulta: {consulta}")
        print("-" * 50)
        for i, r in enumerate(top5, 1):
            meds = r.get("medicamento", "—")
            score = r.get("score", 0)
            texto_excerpt = r["texto"][:100].replace("\n", " ")
            print(f"  {i}. [{score:.3f}] {meds} | {texto_excerpt}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    collection = construir_index(recreate=False)
    demonstrar_buscas(collection)
