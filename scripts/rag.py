from __future__ import annotations

"""Pipeline RAG para Consulta de Interacoes Medicamentosas.

Combina NER (extracao de medicamentos), busca vetorial (ChromaDB),
classificador (BioBERTpt fine-tuned) e LLM (DeepSeek/GPT4All) em um
pipeline end-to-end que retorna JSON estruturado.

Fase 8 — Item: scripts/rag.py (RAGPipeline steps 2-4).
"""

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

from scripts.config import DEVICE, TOP_K_CHUNKS, ANOTACOES_DIR

log = logging.getLogger(__name__)

# ─── Threshold de confianca ──────────────────────────────────────────

DEFAULT_THRESHOLD = 0.3  # confianca minima para acionar LLM generativo

# ─── Prompt template para geracao ────────────────────────────────────

TEMPLATE_RAG = """[CONTEXTO DAS BULAS]
{context_chunks}

[CLASSIFICACAO PRELIMINAR]
O modelo especializado BioBERTpt classificou esta interacao como:
{classe_nome} (confianca: {confianca:.0%})

[INSTRUCAO]
Com base APENAS nas informacoes das bulas acima, explique se ha interacao
entre {medicamento_alvo} e {medicamento_outro}, qual a gravidade e qual
a recomendacao. Se as bulas nao contiverem informacao suficiente, declare
claramente que nao ha dados. Cite a fonte (nome da bula).

[CONSULTA DO USUARIO]
{consulta_original}

[RESPOSTA — Responda em portugues:]"""


# ─── Classes ─────────────────────────────────────────────────────────

class ClassificacaoInteracao:
    """Resultado de uma classificacao individual."""

    def __init__(
        self,
        medicamento_alvo: str,
        medicamento_outro: str,
        classe: int,
        confianca: float,
        classe_nome: str,
        chunks: list[dict],
    ):
        self.medicamento_alvo = medicamento_alvo
        self.medicamento_outro = medicamento_outro
        self.classe = classe
        self.confianca = confianca
        self.classe_nome = classe_nome
        self.chunks = chunks

    def para_dict(self) -> dict:
        return {
            "medicamento_alvo": self.medicamento_alvo,
            "medicamento_outro": self.medicamento_outro,
            "classe": self.classe,
            "classe_nome": self.classe_nome,
            "confianca": round(self.confianca, 4),
            "chunks": self.chunks,
        }


class ResultadoConsulta:
    """Resultado completo de uma consulta ao pipeline RAG."""

    def __init__(
        self,
        query: str,
        sucesso: bool,
        classificacoes: list[ClassificacaoInteracao],
        erro: str | None = None,
    ):
        self.query = query
        self.sucesso = sucesso
        self.classificacoes = classificacoes
        self.erro = erro

    def para_dict(self) -> dict:
        return {
            "query": self.query,
            "sucesso": self.sucesso,
            "n_pares": len(self.classificacoes),
            "classificacoes": [c.para_dict() for c in self.classificacoes],
            "erro": self.erro,
        }

    def para_json(self) -> str:
        return json.dumps(self.para_dict(), ensure_ascii=False, indent=2)


# ─── Funcoes de suporte ─────────────────────────────────────────────

def sanitizar_query(query: str) -> str:
    """Sanitiza a query do usuario contra prompt injection.

    Remove comandos que tentam ignorar instrucoes,
    trunca em 200 caracteres e escapa chaves/colchetes.
    """
    # Padrões de injection conhecidos
    injection_patterns = [
        r"(?i)(ignore|desconsidere|disregard)\s+(todas?|all|everything)",
        r"(?i)(system:|você agora é|you are now|<\|im_start\|>|instruções anteriores)",
        r"(?i)(override|hack|jailbreak|prompt)",
    ]
    for pattern in injection_patterns:
        query = re.sub(pattern, "", query)

    # Remover chaves e colchetes (podem quebrar templates)
    query = query.replace("{", "").replace("}", "").replace("[", "").replace("]", "")

    # Truncar
    return query[:200].strip()


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


def buscar_chunks_relevantes(
    medicamento_alvo: str,
    medicamento_outro: str,
    collection,
    embedder,
    top_k: int = TOP_K_CHUNKS,
    threshold: float = 0.6,
) -> list[dict]:
    """Busca chunks relevantes no ChromaDB via busca hibrida.

    Args:
        medicamento_alvo: nome do primeiro medicamento
        medicamento_outro: nome do segundo medicamento
        collection: ChromaDB collection
        embedder: SentenceTransformer
        top_k: numero de chunks a recuperar
        threshold: distancia maxima (L2 normalizado, < threshold = relevante)

    Returns:
        Lista de dicts com id, texto, medicamento, fonte, distancia
    """
    from scripts.embeddings import buscar_chunks, busca_hibrida

    query = f"{medicamento_alvo} {medicamento_outro} interação"
    emb = embedder.encode([query], normalize_embeddings=True)[0]

    # Tentar busca hibrida primeiro
    try:
        resultados = busca_hibrida(collection, query, emb, n=top_k * 2, alpha=0.3)
    except Exception:
        # Fallback para busca semantica pura
        resultados = buscar_chunks(collection, emb, n=top_k * 2)

    # Filtrar por threshold de distancia
    filtrados = []
    for r in resultados:
        # Distancia L2 normalizada: < 0.6 indica relevancia
        if r.get("distancia", 1.0) < threshold:
            filtrados.append(r)
            if len(filtrados) >= top_k:
                break

    return filtrados[:top_k]


def construir_prompt_rag(
    consulta_original: str,
    medicamento_alvo: str,
    medicamento_outro: str,
    chunks: list[dict],
    classe: int,
    confianca: float,
    classe_nome: str,
) -> str:
    """Constrói o prompt RAG com contexto das bulas.

    Args:
        consulta_original: texto original do usuario
        medicamento_alvo: nome do medicamento principal
        medicamento_outro: nome do segundo medicamento
        chunks: lista de dicts com texto e fonte dos chunks
        classe: classe predita (0, 1, 2)
        confianca: confianca da classificacao (0-1)
        classe_nome: nome da classe

    Returns:
        String com prompt formatado
    """
    # Formatar contexto dos chunks
    contexto_parts = []
    seen_texts: set[str] = set()
    for i, chunk in enumerate(chunks[:5], 1):  # max 5 chunks
        texto = chunk.get("texto", "")[:400].replace("\n", " ").strip()
        fonte = chunk.get("medicamento", "bula desconhecida")
        # Evitar duplicatas
        if texto and texto not in seen_texts:
            seen_texts.add(texto)
            contexto_parts.append(
                f"--- Bula {i} ({fonte}) ---\n{texto}"
            )

    contexto = "\n\n".join(contexto_parts) if contexto_parts else "Sem informacao relevante encontrada nas bulas."

    return TEMPLATE_RAG.format(
        consulta_original=consulta_original,
        medicamento_alvo=medicamento_alvo,
        medicamento_outro=medicamento_outro,
        context_chunks=contexto,
        classe_nome=classe_nome,
        confianca=confianca,
    )


def gerar_resposta_rag(prompt: str, llm_provider) -> str:
    """Gera resposta usando o LLM com contexto RAG.

    Args:
        prompt: prompt ja formatado com contexto
        llm_provider: instancia de LLMProvider

    Returns:
        Texto da resposta gerada
    """
    try:
        resposta = llm_provider.generate(prompt, max_tokens=300, temperature=0.1)
        return resposta.strip() if resposta else "Erro ao gerar resposta."
    except Exception as e:
        log.error("Erro na geracao RAG: %s", e)
        return f"Erro ao gerar resposta: {e}"


# ─── Classe principal ───────────────────────────────────────────────

class RAGPipeline:
    """Pipeline RAG end-to-end para consulta de interacoes medicamentosas.

    Uso:
        pipeline = RAGPipeline(
            ner_model="Ljy3257/clinicalnerpt-chemical",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            classifier_path="data/modelos_finetuned/biobertpt-interactions",
            llm_provider=llm,
        )
        resultado = pipeline.consultar("Posso tomar Amoxicilina com Metotrexato?")
        print(resultado.para_json())
    """

    CLASSES = ["SEM_INTERACAO", "LEVE_MODERADA", "GRAVE_CONTRAINDICADA"]

    def __init__(
        self,
        ner_model: str = "pucpr/clinicalnerpt-chemical",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        classifier_path: str | Path | None = None,
        chroma_path: str | Path = "data/chroma_db",
        collection_name: str = "bulas_interacoes",
        llm_provider=None,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """Inicializa o pipeline RAG.

        Args:
            ner_model: nome do modelo NER no HF Hub
            embedding_model: nome do modelo SentenceTransformer
            classifier_path: caminho para o checkpoint fine-tuned (None = usa base)
            chroma_path: caminho para o banco ChromaDB
            collection_name: nome da collection no ChromaDB
            llm_provider: instancia de LLMProvider (obrigatorio para geracao)
            device: "cuda" ou "cpu"
            threshold: confianca minima para invocar LLM generativo
        """
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.llm = llm_provider

        # 1. NER
        log.info("Inicializando NER...")
        from scripts.ner import MedicationNER
        self.ner = MedicationNER(ner_model, device=self.device)

        # 2. Embedder
        log.info("Inicializando embedder (%s)...", embedding_model)
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embedding_model, cache_folder="data/modelos_cache")

        # 3. ChromaDB
        log.info("Conectando ao ChromaDB (%s)...", chroma_path)
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(name=collection_name)
        log.info("ChromaDB: %d chunks indexados", self.collection.count())

        # 4. Classificador
        log.info("Carregando classificador (%s)...", classifier_path)
        from scripts.classifier import InteractionClassifier
        self.classifier = InteractionClassifier(model_path=classifier_path, device=self.device)

        log.info("Pipeline RAG inicializado com sucesso.")

    def consultar(self, query: str) -> ResultadoConsulta:
        """Executa a consulta completa no pipeline RAG.

        Args:
            query: consulta em linguagem natural do usuario

        Returns:
            ResultadoConsulta com o resultado da analise
        """
        t0 = time.time()

        # Sanitizar
        query_segura = sanitizar_query(query)
        if not query_segura:
            return ResultadoConsulta(query, False, [], erro="Query vazia apos sanitizacao.")

        log.info("Consultando pipeline: '%s'", query_segura[:80])

        # Step 1: NER
        try:
            pares = self.ner.extrair_pares(query_segura)
        except Exception as e:
            log.error("NER falhou: %s", e)
            return ResultadoConsulta(query, False, [], erro=f"NER falhou: {e}")

        if not pares:
            log.info("Nenhum par de medicamentos encontrado.")
            return ResultadoConsulta(
                query, True, [],
                erro=None,
            )

        # Step 2+3: para cada par — busca + classificacao + geracao
        classificacoes: list[ClassificacaoInteracao] = []
        for alvo, outro in pares:
            try:
                # Busca vetorial
                chunks = buscar_chunks_relevantes(
                    alvo, outro, self.collection, self.embedder,
                    top_k=TOP_K_CHUNKS,
                )

                # Classificacao
                resultado_clf = self.classifier.classificar(alvo, outro, chunks)

                # Construir resultado
                clf = ClassificacaoInteracao(
                    medicamento_alvo=alvo,
                    medicamento_outro=outro,
                    classe=resultado_clf["classe"],
                    confianca=resultado_clf["confianca"],
                    classe_nome=resultado_clf["classe_nome"],
                    chunks=chunks,
                )

                # Geracao LLM se confianca >= threshold
                if self.llm and resultado_clf["confianca"] >= self.threshold:
                    prompt = construir_prompt_rag(
                        query_segura, alvo, outro, chunks,
                        resultado_clf["classe"],
                        resultado_clf["confianca"],
                        resultado_clf["classe_nome"],
                    )
                    resposta_llm = gerar_resposta_rag(prompt, self.llm)
                    # Adiciona resposta ao resultado
                    clf.chunks = [
                        {**c, "resposta_gerada": resposta_llm}
                        for c in clf.chunks
                    ]

                classificacoes.append(clf)

            except Exception as e:
                log.error("Erro processando par (%s, %s): %s", alvo, outro, e)
                # Continua com proximo par

        elapsed = time.time() - t0
        log.info("Consulta processada em %.2fs — %d classificacoes", elapsed, len(classificacoes))

        return ResultadoConsulta(query, True, classificacoes)

    def consultar_json(self, query: str) -> str:
        """Conveniencia: retorna JSON diretamente."""
        return self.consultar(query).para_json()


# ─── Demonstração ───────────────────────────────────────────────────

CONSULTAS_DEMONSTRACAO = [
    # (query, descricao)
    ("Posso tomar Amoxicilina com Metotrexato?", "Interacao grave — metotrexato + penicilina"),
    ("Dipirona e AAS juntos fazem mal?", "Interacao leve/moderada"),
    ("Paracetamol com Amoxicilina, pode?", "Sem interacao"),
    ("Invermectina interage com Dipirona?", "Medicamento nao encontrado"),
    ("Posso beber álcool tomando Paracetamol?", "Entidade nao-medicamento"),
    ("Amoxicilina, Ibuprofeno e Dipirona juntos?", "Multiplos medicamentos — 3 pares"),
    ("AAS Protect com Ibuprofeno é seguro?", "Nome comercial + principio ativo"),
    ("Esses dois remédios juntos fazem mal?", "Consulta ambigua — falta medicamentos"),
]


def demonstrar_pipeline():
    """Demonstra o pipeline com 8 consultas de teste."""
    from scripts.classifier import InteractionClassifier
    from scripts.embeddings import construir_index

    print("Inicializando pipeline RAG...")

    # Verificar ChromaDB
    collection = construir_index(recreate=False)

    # Classificador
    classifier = InteractionClassifier()

    print("\n" + "=" * 70)
    print("RAG PIPELINE — DEMONSTRACAO")
    print("=" * 70)

    #Nota: LLM nao instanciado nesta demonstracao (requer API key)
    for query, desc in CONSULTAS_DEMONSTRACAO:
        print(f"\n>>> {query}")
        print(f"    ({desc})")

        # NER apenas para demonstracao
        from scripts.ner import MedicationNER
        ner = MedicationNER()
        meds = ner.extrair_medicamentos(query)
        pares = ner.extrair_pares(query)
        print(f"    NER: {query[:40]} → {len(meds)} meds, {len(pares)} pares: {pares}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    demonstrar_pipeline()
