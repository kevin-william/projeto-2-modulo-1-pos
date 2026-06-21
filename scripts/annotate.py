"""
Anotacao automatica de pares medicamentosos — Fase 3.

Usa weak supervision (heuristica de palavras-chave) para gerar pares rotulados
da Fonte 2, e seleciona casos ambiguos da Fonte 1 para curadoria manual.

Uso:
    python scripts/annotate.py --chunks data/chunks_bulas.jsonl --output data/anotacoes/
"""

import re
import csv
import json
import argparse
import logging
from pathlib import Path
from collections import Counter

from scripts.config import (
    GRAVE_KEYWORDS,
    LEVE_KEYWORDS,
    SEM_INTERACAO_KEYWORDS,
    CLASSES,
)

log = logging.getLogger(__name__)


# ─── Classificacao Heuristica ──────────────────────────────────────

def classificar_heuristicamente(texto: str) -> tuple[int, float]:
    """
    Classifica um texto de interacao com base em palavras-chave.

    Prioridade: GRAVE > LEVE > SEM_INTERACAO.
    Se multiplos matches na mesma classe, usa o maior score.

    Returns:
        (classe, confianca) — classe 0/1/2, confianca 0.0-1.0.
        (-1, -1.0) se nenhum padrao encontrado.
    """
    texto_lower = texto.lower()
    best_classe = -1
    best_confianca = -1.0

    # Classe 2 (GRAVE) — maior prioridade
    for kw in GRAVE_KEYWORDS:
        if kw in texto_lower:
            # Confianca baseada na "forca" da keyword
            conf = 0.90 if any(w in kw for w in ["fatal", "morte", "contraindicado"]) else 0.80
            if conf > best_confianca:
                best_classe, best_confianca = 2, conf

    # Classe 1 (LEVE_MODERADA)
    for kw in LEVE_KEYWORDS:
        if kw in texto_lower:
            conf = 0.75 if "monitorar" in kw or "ajustar" in kw else 0.65
            if 1 > best_classe or (1 == best_classe and conf > best_confianca):
                best_classe, best_confianca = 1, conf

    # Classe 0 (SEM_INTERACAO) — menor prioridade
    for kw in SEM_INTERACAO_KEYWORDS:
        if kw in texto_lower:
            conf = 0.85
            if 0 > best_classe or (0 == best_classe and conf > best_confianca):
                best_classe, best_confianca = 0, conf

    return (best_classe, best_confianca)


# ─── Geracao de Pares ──────────────────────────────────────────────

def gerar_pares_automaticos(
    chunks: list[dict],
    confidence_threshold: float = 0.65,
) -> list[dict]:
    """
    Gera pares (medicamento_alvo, medicamento_outro, contexto, classe)
    a partir de chunks da Fonte 2 usando heuristica.

    Para cada chunk:
    1. Classifica heuristicamente
    2. Se confianca >= threshold, cria par com medicamento_alvo = chunk["medicamento"]
       e medicamento_outro extraido do texto via regex simples de nomes conhecidos

    Args:
        chunks: Lista de chunks (do JSONL).
        confidence_threshold: Confianca minima para incluir o par.

    Returns:
        Lista de pares anotados.
    """
    pares = []
    seen = set()

    for ch in chunks:
        if ch["fonte"] != "fonte2":
            continue

        texto = ch["texto"]
        classe, confianca = classificar_heuristicamente(texto)

        if confianca < confidence_threshold:
            continue

        medicamento_alvo = ch["medicamento"]
        # Tenta extrair medicamento_outro do texto
        outros = _extrair_medicamentos_mencionados(texto, medicamento_alvo)

        for outro in outros:
            key = (medicamento_alvo, outro, texto[:100])
            if key in seen:
                continue
            seen.add(key)

            pares.append({
                "medicamento_alvo": medicamento_alvo,
                "medicamento_outro": outro,
                "contexto": texto,
                "classe": classe,
                "fonte": "fonte2",
                "origem": "automatica",
                "confianca": round(confianca, 2),
            })

    return pares


# Lista de palavras que NAO sao medicamentos
STOP_DRUGS = {
    "medico", "medicamento", "medicamentos", "remedio", "remedios",
    "farmaco", "farmacos", "droga", "drogas", "principio", "ativos",
    "excipiente", "excipientes", "paciente", "pacientes", "tratamento",
    "dose", "doses", "efeito", "efeitos", "uso", "administracao",
    "sangue", "renal", "hepatica", "gravidez", "lactacao",
    "estudos", "estudo", "casos", "relatos", "dados", "informacao",
    "saude", "doenca", "doencas", "reacao", "reacoes", "risco",
    "interacao", "interacoes", "mg", "ml", "kg", "horas", "dias",
}


def _extrair_medicamentos_mencionados(texto: str, alvo: str) -> list[str]:
    """
    Extrai nomes de medicamentos mencionados no texto, excluindo o alvo.

    Usa regex para capturar palavras capitalizadas e termos tecnicos,
    mais uma lista de nomes de medicamentos frequentes nas bulas.
    """
    texto_lower = texto.lower()
    encontrados = []

    # Lista de medicamentos frequentes para match exato
    frequentes = [
        "varfarina", "acenocumarol", "amoxicilina", "alopurinol",
        "probenecida", "ciclosporina", "eritromicina", "claritromicina",
        "ibuprofeno", "paracetamol", "dipirona", "metformina",
        "omeprazol", "captopril", "losartana", "sinvastatina",
        "atorvastatina", "fluoxetina", "sertralina", "diazepam",
        "clonazepam", "fenitoina", "carbamazepina", "digoxina",
        "metotrexato", "cetoconazol", "fluconazol", "itraconazol",
        "rifampicina", "isoniazida", "fenobarbital", "levotiroxina",
        "insulina", "glibenclamida", "hidroclorotiazida", "furosemida",
        "nifedipino", "anlodipino", "enalapril", "diltiazem",
        "verapamil", "amiodarona", "lidocaina", "morfina",
        "codeina", "tramadol", "gabapentina", "acido", "valproico",
        "haloperidol", "risperidona", "quetiapina", "olanzapina",
        "colchicina", "alopurinol", "cimetidina", "ranitidina",
        "contraceptivos", "anticoagulantes", "corticosteroides",
        "aine", "aines", "ainh", "bra", "ieca", "ibr",
        "niacina", "fibratos", "antiacidos", "colestipol",
    ]

    for nome in frequentes:
        if nome in texto_lower and nome != alvo:
            # Verifica se nao e parte de palavra maior
            pattern = re.compile(r"\b" + re.escape(nome) + r"\b")
            if pattern.search(texto_lower):
                encontrados.append(nome)

    # Deduplica mantendo ordem
    unicos = []
    for e in encontrados:
        if e not in unicos and e not in STOP_DRUGS:
            unicos.append(e)

    return unicos[:5]  # max 5 medicamentos por chunk


# ─── Curadoria Manual ──────────────────────────────────────────────

def selecionar_para_curadoria(
    chunks: list[dict],
    n: int = 500,
) -> list[dict]:
    """
    Seleciona chunks da Fonte 1 para anotacao manual.

    Prioriza:
    - Chunks com multiplas entidades (via heuristica de frequentes)
    - Chunks onde a heuristica teve baixa confianca
    - Diversidade de medicamentos

    Returns:
        Lista de dicts com campos: medicamento, secao, texto, classe_sugerida, confianca.
    """
    candidates = []

    for ch in chunks:
        if ch["fonte"] != "fonte1":
            continue

        texto = ch["texto"]
        classe, confianca = classificar_heuristicamente(texto)

        # Prioriza ambiguos (confianca baixa ou classe -1)
        outros = _extrair_medicamentos_mencionados(texto, ch["medicamento"])
        score = len(outros) * 2 + (1.0 - max(confianca, 0)) * 5

        candidates.append({
            "medicamento": ch["medicamento"],
            "secao": ch.get("secao", ""),
            "texto": texto,
            "classe_sugerida": classe,
            "confianca_heuristica": round(confianca, 2),
            "medicamentos_detectados": outros,
            "score_prioridade": round(score, 1),
        })

    # Ordena por score decrescente e pega top N
    candidates.sort(key=lambda x: x["score_prioridade"], reverse=True)
    return candidates[:n]


# ─── Exportacao ────────────────────────────────────────────────────

def exportar_csv(pares: list[dict], caminho: Path):
    """Exporta pares anotados para CSV."""
    caminho.parent.mkdir(parents=True, exist_ok=True)

    colunas = [
        "medicamento_alvo", "medicamento_outro", "contexto",
        "classe", "fonte", "origem", "confianca",
    ]

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for p in pares:
            row = {k: p.get(k, "") for k in colunas}
            writer.writerow(row)

    log.info("Exportado: %d pares → %s", len(pares), caminho)


def exportar_curadoria(candidates: list[dict], caminho: Path):
    """Exporta candidatos para curadoria manual em CSV."""
    caminho.parent.mkdir(parents=True, exist_ok=True)

    colunas = [
        "medicamento", "secao", "texto", "classe_sugerida",
        "confianca_heuristica", "medicamentos_detectados",
        "score_prioridade", "classe_manual",
    ]

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for c in candidates:
            row = {k: c.get(k, "") for k in colunas}
            # classe_manual fica vazia para preenchimento humano
            row["classe_manual"] = ""
            # Serializa lista como string
            if isinstance(row["medicamentos_detectados"], list):
                row["medicamentos_detectados"] = "; ".join(
                    row["medicamentos_detectados"]
                )
            writer.writerow(row)

    log.info("Exportado: %d candidatos → %s", len(candidates), caminho)


def exportar_balanceados(pares: list[dict], caminho: Path):
    """
    Exporta dataset balanceado (undersample da classe majoritaria)
    e dividido em train/val/test (80/10/10).
    """
    from random import seed, shuffle
    seed(42)

    # Agrupa por classe
    by_class = {0: [], 1: [], 2: []}
    for p in pares:
        by_class[p["classe"]].append(p)

    # Undersample para balancear
    min_count = min(len(by_class[c]) for c in (0, 1, 2) if by_class[c])
    min_count = max(min_count, 10)  # pelo menos 10

    balanced = []
    for c in (0, 1, 2):
        shuffle(by_class[c])
        balanced.extend(by_class[c][:min_count])

    shuffle(balanced)
    n = len(balanced)
    train = balanced[: int(n * 0.8)]
    val = balanced[int(n * 0.8): int(n * 0.9)]
    test = balanced[int(n * 0.9):]

    base = caminho.parent
    for name, data in [("train", train), ("val", val), ("test", test)]:
        exportar_csv(data, base / f"{name}.csv")

    log.info("Dataset balanceado: train=%d val=%d test=%d", len(train), len(val), len(test))

    # Estatisticas
    for name, data in [("train", train), ("val", val), ("test", test)]:
        dist = Counter(p["classe"] for p in data)
        log.info("  %s: %s", name, {CLASSES[k]: v for k, v in dist.items()})


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Anotacao automatica de pares medicamentosos (Fase 3)."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/chunks_bulas.jsonl"),
        help="JSONL com chunks pre-processados",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/anotacoes"),
        help="Diretorio de saida para CSVs",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Threshold de confianca minima (default: 0.65)",
    )
    parser.add_argument(
        "--curadoria",
        type=int,
        default=500,
        help="Numero de candidatos para curadoria (default: 500)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Log detalhado"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Carrega chunks
    log.info("Carregando chunks de %s...", args.chunks)
    chunks = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    f1 = sum(1 for c in chunks if c["fonte"] == "fonte1")
    f2 = sum(1 for c in chunks if c["fonte"] == "fonte2")
    log.info("Chunks: %d total (F1=%d, F2=%d)", len(chunks), f1, f2)

    # Gera pares automaticos (Fonte 2)
    log.info("Gerando pares automaticos da Fonte 2...")
    pares = gerar_pares_automaticos(chunks, args.confidence)
    dist = Counter(p["classe"] for p in pares)
    log.info("Pares gerados: %d (%s)",
             len(pares),
             {CLASSES.get(k, "?"): v for k, v in dist.items()})

    exportar_csv(pares, args.output / "automaticas.csv")

    # Seleciona candidatos para curadoria (Fonte 1)
    log.info("Selecionando %d candidatos para curadoria manual...", args.curadoria)
    candidates = selecionar_para_curadoria(chunks, args.curadoria)
    log.info("Candidatos selecionados: %d", len(candidates))

    exportar_curadoria(candidates, args.output / "pendentes_curadoria.csv")

    # Dataset balanceado
    if pares:
        log.info("Gerando dataset balanceado...")
        exportar_balanceados(pares, args.output / "train.csv")

    log.info("Concluido.")


if __name__ == "__main__":
    main()
