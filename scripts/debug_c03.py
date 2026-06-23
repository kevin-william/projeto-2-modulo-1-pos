"""
Debug script para c03_embeddings_busca.ipynb

Executar:
    cd /c/workspace/python/projeto-2-modulo-1-pos
    source venv/Scripts/activate
    python scripts/debug_c03.py          # todas as células
    python scripts/debug_c03.py 11       # só célula 11
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

import numpy as np
import pandas as pd
import time
import json

from scripts.config import (
    CHUNKS_BULAS, COLLECTION_NAME, CHROMA_DIR,
)
from scripts.embeddings import (
    carregar_chunks, gerar_embeddings, criar_collection,
    indexar_chunks, construir_index, buscar_chunks,
    busca_hibrida,
)


class DebugC03:
    """Debug do c03_embeddings_busca.ipynb

    Células:
      05 - Carregar/pre-processar chunks (ou gerar se não existir)
      09 - Gerar embeddings MINILM
      11 - Construir/indexar collection ChromaDB
      13 - Métricas de busca (P@3, MRR, latência)
      14 - Gráfico comparativo
      16 - Busca híbrida (demo)
      18 - Sweep alpha na busca híbrida
      20 - Demonstração de consultas
    """

    def __init__(self):
        self.df_chunks   = None
        self.embeddings  = None
        self.collection  = None
        self.MODELS = {
            "MINILM": "sentence-transformers/all-MiniLM-L6-v2",
            "BERTPT": "pucpr/biobertpt-all",
            "E5":     "intfloat/e5-base-v2",
        }

    # ── cell 05 ─────────────────────────────────────────────────────────────

    def cell_05(self):
        """Carrega chunks_bulas.jsonl — erro se não existir."""
        from pathlib import Path

        print("[cell_05] Carregando chunks...")
        chunks_path = Path(CHUNKS_BULAS)

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks não existem: {chunks_path}\n"
                "Gere primeiro: python scripts/preprocess.py"
            )
        df = carregar_chunks(CHUNKS_BULAS)
        self.df_chunks = df
        print(f"  Shape: {df.shape}")
        print(f"  Colunas: {list(df.columns)}")
        if "n_chars" in df.columns:
            print(f"  n_chars stats:\n{df['n_chars'].describe()}")

    # ── cell 09 ─────────────────────────────────────────────────────────────

    def cell_09(self):
        """Gera embeddings com MINILM."""
        print("[cell_09] Gerando embeddings MINILM...")
        textos = self.df_chunks["texto"].tolist()

        t0 = time.time()
        emb = gerar_embeddings(textos, modelo_nome="MINILM")
        elapsed = time.time() - t0

        self.embeddings = emb
        print(f"  Shape: {emb.shape} | Tempo: {elapsed:.1f}s")

    # ── cell 11 ─────────────────────────────────────────────────────────────

    def cell_11(self):
        """Constrói (ou recarrega) a collection ChromaDB."""
        print("[cell_11] Construindo index ChromaDB...")
        collection = construir_index(
            chunks_path=CHUNKS_BULAS,
            modelo_embedding="MINILM",
            recreate=True,
        )
        self.collection = collection
        print(f"\nCollection: {collection.name}")
        print(f"Total de documentos indexados: {collection.count()}")

    # ── cell 13 ─────────────────────────────────────────────────────────────

    def cell_13(self):
        """Avaliação de busca — P@3, MRR, latência (só MINILM).

        NOTA: o ChromaDB foi indexado com MINILM (384d). Iterar sobre
        BERTPT (768d) ou E5 (768d) causa InvalidArgumentError.
        Por isso o loop é fixo em ['MINILM'].
        """
        from sentence_transformers import SentenceTransformer

        # Carrega collection se não existe (independente de cell_11)
        if self.collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.collection = client.get_collection(COLLECTION_NAME)

        print(f"  Collection: {self.collection.count()} docs")

        CONSULTAS_TESTE = [
            # IDs já em lowercase (normalizados na indexação ChromaDB)
            ("sinvastatina itraconazol interação contraindicado cyp3a4",
             ["f1_100470472_sinvastatina_profissional_005",
              "f1_100431188_itraconazol_profissional_003"]),
            ("amoxicilina metotrexato penicilina toxicidade",
             ["f1_100431004_amoxicilina_clavulanato_de_potássio_profissional_009"]),
            ("metformina insuficiência renal hipóxia",
             ["f1_100470663_fosfato_de_sitagliptina_cloridrato_de_metformina_profissional_007"]),
            ("varfarina sangramento anticoagulante",
             ["f1_103700512_varfarina_sódica_paciente_000"]),
            ("ibuprofeno paracetamol interação",
             ["f1_100431469_ibuprofeno_paracetamol_paciente_000"]),
            ("losartana potássio hipercalemia",
             ["f1_100430911_losartana_potássica_profissional_011"]),
            ("omeprazol claritromicina interação cyp3a4",
             ["f1_102351182_esomeprazol_magnésico_tri-hidratado_profissional_005"]),
            ("fluoxetina tramadol síndrome serotoninérgica",
             ["f1_102350464_cloridrato_de_fluoxetina_profissional_020"]),
            ("atorvastatina interagir medicamento",
             ["f1_100431137_atorvastatina_cálcica_profissional_000"]),
            ("dexametasona dipirona interação",
             ["f1_100431331_fosfato_dissódico_de_dexametasona_profissional_000"]),
        ]

        def metricas_busca(collection, modelo_key, consultas, n=3):
            model = SentenceTransformer(
                self.MODELS[modelo_key],
                cache_folder="data/modelos_cache",
            )
            p_scores, mrr_scores, latencias = [], [], []

            for consulta, ids_esperados in consultas:
                t0 = time.time()
                emb = model.encode([consulta], normalize_embeddings=True)[0]
                top_n = buscar_chunks(collection, emb, n=n)
                lat = (time.time() - t0) * 1000

                ids_ret = [r["id"] for r in top_n]
                p = len(set(ids_ret) & set(ids_esperados)) / n
                mrr = 0.0
                for rank, rid in enumerate(ids_ret, 1):
                    if rid in ids_esperados:
                        mrr = 1.0 / rank
                        break

                p_scores.append(p)
                mrr_scores.append(mrr)
                latencias.append(lat)

            return {
                "modelo": modelo_key,
                "dims": model.get_sentence_embedding_dimension(),
                "p_at_3": round(np.mean(p_scores), 3),
                "mrr": round(np.mean(mrr_scores), 3),
                "lat_ms": round(np.mean(latencias), 1),
            }

        resultados = []

        # ATENÇÃO: só MINILM — ChromaDB foi indexado com embedding 384d
        for nome in ["MINILM"]:
            r = metricas_busca(self.collection, nome, CONSULTAS_TESTE)
            resultados.append(r)
            print(f"  {r['modelo']}: P@3={r['p_at_3']} | MRR={r['mrr']} "
                  f"| Lat={r['lat_ms']}ms | {r['dims']}d")

        df_result = pd.DataFrame(resultados)
        print("\nResumo:")
        print(df_result)

    # ── cell 14 ─────────────────────────────────────────────────────────────

    def cell_14(self):
        """Gráfico comparativo (barras P@3 e MRR)."""
        print("[cell_14] cell_13 precisa rodar antes — gráficos omitidos no debug")
        # O gráfico requer matplotlib — pula no debug script
        print("  (gráfico dispensável para validação de lógica)")

    # ── cell 16 ─────────────────────────────────────────────────────────────

    def cell_16(self):
        """Demonstração de busca híbrida com uma consulta."""
        print("[cell_16] Demo — busca híbrida...")

        QUERY = "sinvastatina itraconazol interação"
        modelo = self.MODELS["MINILM"]
        n = 5

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.MODELS["MINILM"],
                                     cache_folder="data/modelos_cache")
        QUERY = "sinvastatina itraconazol interação"
        emb = model.encode([QUERY], normalize_embeddings=True)[0]

        resultados = busca_hibrida(
            self.collection, QUERY, emb,
            n=5, alpha=0.7,
        )

        for i, r in enumerate(resultados, 1):
            print(f"  {i}. [{r['score']:.3f}] cos={r['cos_score']:.3f} "
                  f"bm25={r['bm25_score']:.1f} | {r['medicamento'][:30]}")
            print(f"     {r['texto'][:120]}...")

    # ── cell 18 ─────────────────────────────────────────────────────────────

    def cell_18(self):
        """Sweep de alpha na busca híbrida."""
        print("[cell_18] Sweep alpha...")

        QUERY = "metformina insuficiencia renal"
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        modelo = self.MODELS["MINILM"]

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(modelo, cache_folder="data/modelos_cache")
        emb = model.encode([QUERY], normalize_embeddings=True)[0]

        for alpha in alphas:
            resultados = busca_hibrida(
                self.collection, QUERY, emb,
                n=5, alpha=alpha,
            )
            top_id = resultados[0]["id"] if resultados else "n/a"
            top_score = resultados[0]["score"] if resultados else 0
            print(f"  alpha={alpha:.1f} → top_id={top_id} score={top_score:.3f}")

        print("  (alpha=1.0 = só vetorial, alpha=0.0 = só BM25)")

    # ── cell 20 ─────────────────────────────────────────────────────────────

    def cell_20(self):
        """Demonstração de consultas médicas."""
        print("[cell_20] Demonstração de consultas...")

        CONSULTAS_DEMONSTRACAO = [
            ("amoxicilina com ibuprofeno", "antibiótico e anti-inflamatório"),
            ("metformina e insuficiência renal", "paciente diabético renal"),
            ("warfarina e aspirina", "anticoagulante e antiagregante"),
        ]

        for texto, ctx in CONSULTAS_DEMONSTRACAO:
            QUERY = f"{texto} — {ctx}"
            print(f"\n  Query: {QUERY}")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                self.MODELS["MINILM"],
                cache_folder="data/modelos_cache",
            )
            emb = model.encode([QUERY], normalize_embeddings=True)[0]
            top = buscar_chunks(self.collection, emb, n=3)
            for i, r in enumerate(top, 1):
                print(f"    {i}. [{r['dist']:.3f}] {r['medicamento']} | {r['texto'][:100]}...")

    # ── run helpers ─────────────────────────────────────────────────────────

    def run_all(self):
        """Executa todas as células em ordem (simula 'Run All')."""
        print("=" * 60)
        print("run_all — c03_embeddings_busca")
        print("=" * 60)
        self.cell_05()
        self.cell_09()
        self.cell_11()
        self.cell_13()
        self.cell_14()
        self.cell_16()
        self.cell_18()
        self.cell_20()
        print("\n✓ run_all completo")

    def run_cell(self, n):
        """Executa uma célula específica pelo número."""
        getattr(self, f"cell_{n:02d}")()


if __name__ == "__main__":
    dbg = DebugC03()

    if len(sys.argv) > 1:
        # python debug_c03.py 11
        for arg in sys.argv[1:]:
            dbg.run_cell(int(arg))
    else:
        dbg.run_all()
