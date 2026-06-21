"""
Testes do modulo de pre-processamento de bulas (Fase 1).

Cobre: classificacao de fonte, extracao de medicamento, chunking,
parseamento de Fonte 1 e Fonte 2.
"""

import textwrap
from pathlib import Path

import pytest

from scripts.preprocess import (
    classificar_fonte,
    extrair_medicamento_alvo,
    chunk_em_sentencas,
    extrair_secoes_fonte1,
    extrair_bloco_interacao_fonte2,
    processar_bulas,
)


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def amostra_fonte1_profissional():
    """Trecho real de bula ANVISA profissional com interacoes."""
    return textwrap.dedent("""\
    ## CONTRAINDICACOES

    4. CONTRAINDICACOES
    A amoxicilina e uma penicilina e nao deve ser administrada a pacientes com
    historico de hipersensibilidade aos antibioticos betalactamicos.

    ## INTERACOES  MEDICAMENTOSAS

    6. INTERACOES  MEDICAMENTOSAS
    A probenecida reduz a secrecao tubular renal da amoxicilina. No uso
    concomitante com amoxicilina, pode haver aumento dos niveis de amoxicilina
    no sangue e no prolongamento dessa alteracao.

    A administracao concomitante de alopurinol durante o tratamento com
    amoxicilina pode aumentar a probabilidade de reacoes alergicas da pele.

    ## REACOES ADVERSAS

    9. REACOES ADVERSAS
    As reacoes adversas estao organizadas segundo frequencia e sistemas.
    """)


@pytest.fixture
def amostra_fonte1_paciente():
    """Trecho de bula ANVISA paciente com secao 'O QUE DEVO SABER ANTES DE USAR'."""
    return textwrap.dedent("""\
    ## O QUE DEVO SABER ANTES DE USAR

    Informe ao seu medico se voce esta tomando outros medicamentos,
    especialmente anticoagulantes como varfarina. O uso concomitante pode
    aumentar o risco de sangramento.

    ## COMO DEVO USAR

    Tome este medicamento conforme orientacao medica.
    """)


@pytest.fixture
def amostra_fonte1_sem_secao():
    """Bula Fonte 1 sem nenhuma secao de interesse."""
    return textwrap.dedent("""\
    ## IDENTIFICACAO

    Medicamento: Placebo 500 mg.

    ## DIZERES LEGAIS

    Registro: 1.2345.6789
    """)


@pytest.fixture
def amostra_fonte2():
    """Trecho real de bula Consultaremedios (zarator)."""
    return textwrap.dedent("""\
    [P: PRECAUCOES?]
    R: Quais cuidados devo ter ao usar o Zarator? Deve ser usado com cuidado...

    [P: INTERACAO MEDICAMENTOSA?]
    R: Interacao medicamentosa: quais os efeitos de tomar Zarator com outros
    remedios? Miopatia pode ocorrer em pacientes que usam Zarator, sendo mais
    frequentes naqueles que usam tambem ciclosporina, fibratos, niacina ou
    antifungicos azolicos. A administracao concomitante de Zarator com
    medicamentos inibidores do citocromo P450 3A4 (por ex., ciclosporina,
    eritromicina/claritromicina, inibidores da protease) pode alterar a
    quantidade de atorvastatina no sangue.
    """)


@pytest.fixture
def tmp_dir(tmp_path):
    """Cria estrutura de diretorios com amostras de bulas."""
    f1 = tmp_path / "fonte1"
    f2 = tmp_path / "fonte2"
    f1.mkdir()
    f2.mkdir()

    # Cria arquivos de amostra
    (f1 / "105830895_amoxicilina_profissional.txt").write_text(
        textwrap.dedent("""\
        ## INTERACOES  MEDICAMENTOSAS
        A probenecida reduz a secrecao tubular renal da amoxicilina.
        A administracao concomitante de alopurinol pode aumentar reacoes.
        """)
    )
    (f2 / "zarator.txt").write_text(
        textwrap.dedent("""\
        [P: INTERACAO MEDICAMENTOSA?]
        R: Miopatia pode ocorrer com ciclosporina. A administracao concomitante
        com eritromicina pode alterar a quantidade de atorvastatina no sangue.
        """)
    )
    return tmp_path


# ─── Testes: Classificacao ─────────────────────────────────────────

class TestClassificarFonte:
    def test_fonte1_por_prefixo_numerico(self):
        """Arquivos comecando com digito + underscore sao Fonte 1."""
        assert classificar_fonte("105830895_amoxicilina_profissional.txt") == "fonte1"

    def test_fonte2_por_letra(self):
        """Arquivos comecando com letra sao Fonte 2."""
        assert classificar_fonte("zarator.txt") == "fonte2"

    def test_fonte1_paciente(self):
        """Versao paciente da Fonte 1 tambem e classificada corretamente."""
        assert classificar_fonte("100380097_secnidazol_paciente.txt") == "fonte1"


# ─── Testes: Extracao de Medicamento ───────────────────────────────

class TestExtrairMedicamentoAlvo:
    def test_fonte1_profissional(self):
        """Extrai nome de arquivo Fonte 1 profissional."""
        assert extrair_medicamento_alvo("105830895_amoxicilina_profissional.txt") == "amoxicilina"

    def test_fonte1_paciente(self):
        """Extrai nome de arquivo Fonte 1 paciente."""
        assert extrair_medicamento_alvo("100380098_captopril_paciente.txt") == "captopril"

    def test_fonte2(self):
        """Extrai nome de arquivo Fonte 2."""
        assert extrair_medicamento_alvo("zarator.txt") == "zarator"

    def test_nome_composto(self):
        """Medicamento com nome composto (underscores → espacos)."""
        resultado = extrair_medicamento_alvo(
            "123520273_amoxicilina_clavulanato_de_potassio_profissional.txt"
        )
        assert "amoxicilina" in resultado
        assert "clavulanato" in resultado


# ─── Testes: Chunking ──────────────────────────────────────────────

class TestChunkEmSentencas:
    def test_split_basico(self):
        """Divide texto em sentencas por pontuacao."""
        texto = "O medicamento amoxicilina e seguro para uso pediatrico. Nao ha interacoes clinicamente relevantes conhecidas. Consulte seu medico sobre contraindicacoes."
        resultado = chunk_em_sentencas(texto)
        assert len(resultado) == 3

    def test_filtra_sentenca_curta(self):
        """Sentencas menores que min_chars sao descartadas."""
        texto = "OK. O medicamento e seguro e pode ser usado conforme prescricao medica."
        resultado = chunk_em_sentencas(texto, min_chars=30)
        assert len(resultado) == 1  # "OK." descartada

    def test_filtra_sentenca_longa(self):
        """Sentencas com mais de max_palavras sao descartadas."""
        palavras = ["palavra"] * 300
        texto = " ".join(palavras) + "."
        resultado = chunk_em_sentencas(texto, max_palavras=250)
        assert len(resultado) == 0

    def test_texto_sem_pontuacao(self):
        """Texto sem pontuacao e retornado como unica sentenca se dentro dos limites."""
        texto = "O medicamento e seguro e pode ser usado conforme prescricao medica recomendada"
        resultado = chunk_em_sentencas(texto)
        assert len(resultado) == 1


# ─── Testes: Parseamento Fonte 1 ───────────────────────────────────

class TestExtrairSecoesFonte1:
    def test_profissional_com_interacoes(self, amostra_fonte1_profissional):
        """Extrai multiplas secoes de interesse da bula profissional."""
        secoes = extrair_secoes_fonte1(amostra_fonte1_profissional)
        nomes = [s["secao"] for s in secoes]
        assert len(secoes) >= 2
        assert any("CONTRAINDICACOES" in n.upper() for n in nomes)
        assert any("INTERACOES" in n.upper() for n in nomes)

    def test_paciente_com_fallback(self, amostra_fonte1_paciente):
        """Extrai secao 'O QUE DEVO SABER ANTES DE USAR' como fallback."""
        secoes = extrair_secoes_fonte1(amostra_fonte1_paciente)
        assert len(secoes) == 1
        assert "SABER" in secoes[0]["secao"].upper()

    def test_sem_secao_relevante(self, amostra_fonte1_sem_secao):
        """Bula sem secoes de interesse retorna lista vazia."""
        secoes = extrair_secoes_fonte1(amostra_fonte1_sem_secao)
        assert secoes == []


# ─── Testes: Parseamento Fonte 2 ───────────────────────────────────

class TestExtrairBlocoFonte2:
    def test_extrai_bloco_interacao(self, amostra_fonte2):
        """Extrai o bloco INTERACAO MEDICAMENTOSA? da Fonte 2."""
        conteudo = extrair_bloco_interacao_fonte2(amostra_fonte2)
        assert conteudo is not None
        assert "ciclosporina" in conteudo.lower()

    def test_sem_bloco_interacao(self):
        """Arquivo sem bloco INTERACAO retorna None."""
        texto = "[P: COMPOSICAO?]\nR: Principio ativo: atorvastatina."
        conteudo = extrair_bloco_interacao_fonte2(texto)
        assert conteudo is None


# ─── Testes: Integracao ────────────────────────────────────────────

class TestProcessarBulas:
    def test_smoke_test(self, tmp_dir):
        """Processa 2 arquivos de amostra e verifica saida JSONL."""
        output = tmp_dir / "chunks.jsonl"
        stats = processar_bulas(tmp_dir, output)

        assert stats["total_arquivos"] == 2
        assert stats["total_chunks"] >= 2
        assert stats["erros"] == 0
        assert output.exists()

        # Verifica que o JSONL e valido
        with open(output, "r", encoding="utf-8") as f:
            for line in f:
                chunk = __import__("json").loads(line)
                assert "id" in chunk
                assert "medicamento" in chunk
                assert "fonte" in chunk
                assert "texto" in chunk
                assert "tokens" in chunk
