"""
Pre-processamento das bulas — Fase 1.

Parseia seletivamente bulas da Fonte 1 (ANVISA) e Fonte 2 (Consultaremedios),
extrai secoes de interacoes medicamentosas, divide em sentencas,
e salva em JSONL para indexacao futura.

Uso:
    python scripts/preprocess.py --input data/bulas --output data/chunks_bulas.jsonl
"""

import re
import json
import argparse
import logging
from pathlib import Path

from scripts.config import (
    FONTE1_PATTERN,
    SECOES_MANTER_F1,
    BLOCOS_MANTER_F2,
    CHARS_PER_TOKEN,
)

log = logging.getLogger(__name__)


# ─── Classificacao de Fonte ───────────────────────────────────────

def classificar_fonte(nome_arquivo: str) -> str:
    """
    Retorna 'fonte1' se o nome comeca com digito + underscore,
    'fonte2' caso contrario.

    Exemplos:
        '105830895_amoxicilina_profissional.txt' → 'fonte1'
        'zarator.txt' → 'fonte2'
    """
    base = Path(nome_arquivo).stem
    return "fonte1" if FONTE1_PATTERN.match(base) else "fonte2"


def extrair_medicamento_alvo(nome_arquivo: str) -> str:
    """
    Extrai o nome do medicamento do nome do arquivo.

    Fonte 1: '105830895_amoxicilina_profissional.txt' → 'amoxicilina'
    Fonte 2: 'zarator.txt' → 'zarator'

    Remove prefixo numerico, sufixo de versao, underscores → espacos.
    """
    base = Path(nome_arquivo).stem

    # Fonte 1: remove prefixo numerico
    base = re.sub(r"^\d+_", "", base)

    # Remove sufixos de versao
    base = re.sub(r"_(paciente|profissional)$", "", base, flags=re.IGNORECASE)

    # Normaliza: underscores → espacos, lowercase, strip
    medicamento = base.replace("_", " ").strip().lower()
    return medicamento


# ─── Chunking ──────────────────────────────────────────────────────

# Regex para split de sentencas (pontuacao seguida de espaco + maiuscula ou fim)
SENTENCA_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ú\(])")


def chunk_em_sentencas(
    texto: str, min_chars: int = 30, max_palavras: int = 250
) -> list[str]:
    """
    Divide texto em sentencas e filtra por tamanho.

    Args:
        texto: Texto a ser dividido.
        min_chars: Tamanho minimo em caracteres (sentencas menores sao descartadas).
        max_palavras: Tamanho maximo em palavras (sentencas maiores sao descartadas).

    Returns:
        Lista de sentencas.
    """
    # Pre-limpeza: normaliza espacos e quebras de linha
    texto = re.sub(r"\s+", " ", texto).strip()

    # Split por pontuacao
    partes = SENTENCA_SPLIT.split(texto)

    # Se o split nao funcionou (texto sem pontuacao), retorna o texto inteiro
    if len(partes) <= 1 and len(texto) >= min_chars:
        palavras = len(texto.split())
        return [texto] if palavras <= max_palavras else []

    sentencas = []
    for s in partes:
        s = s.strip()
        if len(s) < min_chars:
            continue
        palavras = len(s.split())
        if palavras > max_palavras:
            continue
        sentencas.append(s)

    return sentencas


# ─── Parsers de Fonte ──────────────────────────────────────────────

# Secoes da Fonte 1 delimitadas por '## SECAO'
SECAO_F1_REGEX = re.compile(r"##\s*([^\n]+)\s*\n(.*?)(?=\n##|\Z)", re.DOTALL)

# Blocos da Fonte 2: [P: PERGUNTA?] R: resposta
BLOCO_F2_REGEX = re.compile(
    r"\[P:\s*([^\]]+)\]\s*\nR:\s*(.*?)(?=\n\[P:|\Z)", re.DOTALL
)


def _normalizar_secao(nome: str) -> str:
    """Normaliza nome de secao para matching: lowercase, remove acentos, trim."""
    import unicodedata
    nome = nome.strip().lower()
    # Remove acentos (NFC → NFD → remove combining chars)
    nome = unicodedata.normalize('NFD', nome)
    nome = ''.join(c for c in nome if unicodedata.category(c) != 'Mn')
    return nome


def _fuzzy_match_secao(nome_secao: str, lista_alvo: set[str], threshold: int = 3) -> bool:
    """
    Verifica se uma secao esta na lista-alvo, com tolerancia a erros.
    Usa correspondencia exata + substring como fallback.
    """
    norm = _normalizar_secao(nome_secao)
    # Match exato
    if norm in lista_alvo:
        return True
    # Substring
    for alvo in lista_alvo:
        if alvo in norm or norm in alvo:
            return True
    return False


def extrair_secoes_fonte1(texto: str) -> list[dict]:
    """
    Extrai secoes de interesse da Fonte 1 (bulas ANVISA).

    Retorna lista de dicts: {'secao': str, 'conteudo': str}
    """
    secoes = []
    for match in SECAO_F1_REGEX.finditer(texto):
        nome = match.group(1).strip()
        conteudo = match.group(2).strip()
        if _fuzzy_match_secao(nome, SECOES_MANTER_F1) and conteudo:
            secoes.append({"secao": nome, "conteudo": conteudo})
    return secoes


def extrair_bloco_interacao_fonte2(texto: str) -> str | None:
    """
    Extrai o bloco INTERACAO MEDICAMENTOSA? da Fonte 2 (Consultaremedios).

    Returns:
        Conteudo do bloco, ou None se nao encontrado.
    """
    import unicodedata

    def _sem_acentos(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    for match in BLOCO_F2_REGEX.finditer(texto):
        nome = _sem_acentos(match.group(1).strip().upper())
        if "INTERACAO" in nome and "MEDICAMENTOSA" in nome:
            return match.group(2).strip()
    return None


# ─── Pipeline Principal ────────────────────────────────────────────

def processar_bulas(
    data_dir: Path, output_path: Path, max_files: int | None = None
) -> dict:
    """
    Processa todas as bulas e salva chunks em JSONL.

    Args:
        data_dir: Diretorio raiz contendo fonte1/ e fonte2/.
        output_path: Caminho do arquivo JSONL de saida.
        max_files: Limite de arquivos para smoke test (None = todos).

    Returns:
        Dict com estatisticas.
    """
    stats = {
        "total_arquivos": 0,
        "total_chunks": 0,
        "fonte1": 0,
        "fonte2": 0,
        "sem_secao": 0,
        "erros": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for fonte_dir, fonte_nome in [
            (data_dir / "fonte1", "fonte1"),
            (data_dir / "fonte2", "fonte2"),
        ]:
            if not fonte_dir.is_dir():
                log.warning("Diretorio nao encontrado: %s", fonte_dir)
                continue

            arquivos = sorted(fonte_dir.glob("*.txt"))
            if max_files:
                arquivos = arquivos[:max_files]

            for arq in arquivos:
                stats["total_arquivos"] += 1
                try:
                    chunks = _processar_arquivo(arq, fonte_nome)
                    if chunks:
                        for ch in chunks:
                            f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                            stats["total_chunks"] += 1
                    else:
                        stats["sem_secao"] += 1
                    stats[fonte_nome] += 1
                except Exception as e:
                    log.error("Erro ao processar %s: %s", arq.name, e)
                    stats["erros"] += 1

    return stats


def _processar_arquivo(caminho: Path, fonte: str) -> list[dict]:
    """Processa um arquivo de bula e retorna lista de chunks."""
    nome_arquivo = caminho.name
    medicamento = extrair_medicamento_alvo(nome_arquivo)

    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()

    if fonte == "fonte1":
        secoes = extrair_secoes_fonte1(texto)
        if not secoes:
            return []
        chunks = []
        for sec in secoes:
            sentencas = chunk_em_sentencas(sec["conteudo"])
            for i, sent in enumerate(sentencas):
                chunks.append({
                    "id": f"f1_{caminho.stem}_{i:03d}",
                    "medicamento": medicamento,
                    "fonte": "fonte1",
                    "secao": sec["secao"],
                    "texto": sent,
                    "tokens": _estimar_tokens(sent),
                })
        return chunks

    else:  # fonte2
        conteudo = extrair_bloco_interacao_fonte2(texto)
        if not conteudo:
            return []
        sentencas = chunk_em_sentencas(conteudo)
        return [
            {
                "id": f"f2_{caminho.stem}_{i:03d}",
                "medicamento": medicamento,
                "fonte": "fonte2",
                "secao": "INTERACAO MEDICAMENTOSA",
                "texto": sent,
                "tokens": _estimar_tokens(sent),
            }
            for i, sent in enumerate(sentencas)
        ]


def _estimar_tokens(texto: str) -> int:
    """Estimativa grosseira de tokens (1 token ~= 4 caracteres em portugues)."""
    return max(1, len(texto) // CHARS_PER_TOKEN)


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-processa bulas e gera chunks JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bulas"),
        help="Diretorio com fonte1/ e fonte2/ (default: data/bulas)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks_bulas.jsonl"),
        help="Arquivo JSONL de saida (default: data/chunks_bulas.jsonl)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limite de arquivos por fonte (smoke test)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Log detalhado"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    log.info("Iniciando pre-processamento...")
    log.info("Input:  %s", args.input.resolve())
    log.info("Output: %s", args.output.resolve())

    stats = processar_bulas(args.input, args.output, args.max_files)

    log.info("=== Resumo ===")
    log.info("Arquivos processados: %d", stats["total_arquivos"])
    log.info("  Fonte 1: %d  |  Fonte 2: %d", stats["fonte1"], stats["fonte2"])
    log.info("Chunks gerados:      %d", stats["total_chunks"])
    log.info("Sem secao relevante:  %d", stats["sem_secao"])
    log.info("Erros:               %d", stats["erros"])
    log.info("Saida: %s", args.output.resolve())


if __name__ == "__main__":
    main()
