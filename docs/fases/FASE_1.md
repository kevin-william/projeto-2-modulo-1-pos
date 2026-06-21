# Fase 1: Pré-processamento das Bulas

**Objetivo:** Parsear seletivamente as 5.960 bulas (Fonte 1 + Fonte 2), extrair apenas as seções de interações medicamentosas, dividir em sentenças, e salvar em JSONL para indexação futura.

**Dependências:** Fase 0 concluída (config.py, PyTorch CUDA).

---

## Estratégia de Parseamento

### Fonte 1 (ANVISA) — Texto corrido

1. Detectar versão pelo sufixo (`_paciente` ou `_profissional`)
2. Extrair seções prioritárias via regex:
   - **Profissional:** "INTERAÇÕES MEDICAMENTOSAS" → fallback: "PRECAUÇÕES", "CONTRAINDICAÇÕES"
   - **Paciente:** "INTERAÇÕES MEDICAMENTOSAS" → fallback: "O QUE DEVO SABER ANTES DE USAR?", "REAÇÕES ADVERSAS"
3. Extrair medicamento-alvo do nome do arquivo
4. Descartar o restante da bula

### Fonte 2 (Consultaremedios) — Q&A estruturado

1. Identificar bloco `10. INTERAÇÃO MEDICAMENTOSA?` via regex
2. Extrair resposta completa do bloco
3. Extrair também bloco `3. COMPOSIÇÃO?` (para excipientes)
4. Split da resposta em sentenças por pontuação

---

## Tarefas

### 1.1 Criar `scripts/preprocess.py`

- [x] Função `classificar_fonte(nome_arquivo: str) -> str` — retorna `"fonte1"` ou `"fonte2"`
  - Fonte 1: nome começa com dígito + `_`
  - Fonte 2: nome começa com letra + `_`
- [x] Função `extrair_secoes_fonte1(texto: str, versao: str) -> str`
  - Regex para capturar seções entre cabeçalhos em MAIÚSCULAS
  - Fuzzy match contra lista de seções-alvo (threshold Levenshtein = 3)
  - Concatena seções encontradas com marcador `[SEÇÃO: nome]`
- [x] Função `extrair_bloco_fonte2(texto: str) -> str`
  - Regex: `r"10\.\s*INTERAÇÃO\s*MEDICAMENTOSA\?[^\d]*(?=\d+\.|\Z)"`
  - Fallback: busca por substring "INTERAÇÃO MEDICAMENTOSA"
- [x] Função `extrair_medicamento_alvo(nome_arquivo: str) -> str`
  - Remove prefixo numérico (Fonte 1) e sufixo de versão
  - Normaliza: lowercase, strip, underscores → espaços

### 1.2 Criar função de chunking

- [x] Função `chunk_em_sentencas(texto: str) -> list[str]`
  - Split por `.`, `?`, `!`, `;`
  - Filtra: mínimo 30 caracteres, máximo 250 palavras
  - Preserva metadados: `medicamento`, `fonte`, `secao_original`
- [x] Testar com amostra: sentença muito curta ("Veja.") → descartada; parágrafo longo → split correto

### 1.3 Pipeline principal

- [x] Função `processar_bulas(data_dir: Path, output_path: Path) -> dict`
  - Itera sobre `fonte1/` e `fonte2/`
  - Para cada arquivo: classifica fonte → extrai seções → chunk → adiciona ao JSONL
  - Registra estatísticas: quantos arquivos processados, quantos sem seção de interação, chunks gerados
  - Barra de progresso com `tqdm`
- [x] Salvar `data/chunks_bulas.jsonl` com estrutura:
  ```json
  {"id": "f1_12345_amoxicilina_001", "medicamento": "amoxicilina", "fonte": "fonte1",
   "secao": "INTERAÇÕES MEDICAMENTOSAS", "texto": "A administração concomitante...", "tokens": 42}
  ```

### 1.4 Testes

- [x] `tests/test_preprocess.py`:
  - [x] `test_classificar_fonte_f1` — arquivo `12345_medicamento_profissional.txt` → `"fonte1"`
  - [x] `test_classificar_fonte_f2` — arquivo `medicamento-profissional.txt` → `"fonte2"`
  - [x] `test_extrair_secoes_fonte1_profissional` — amostra com seção "INTERAÇÕES MEDICAMENTOSAS"
  - [x] `test_extrair_secoes_fonte1_paciente` — amostra com seção "O QUE DEVO SABER ANTES DE USAR?"
  - [x] `test_extrair_secoes_fonte1_sem_secao` — bula sem seção de interação → fallback funciona
  - [x] `test_extrair_bloco_fonte2` — bloco 10 extraído corretamente
  - [x] `test_extrair_medicamento_alvo` — nome extraído e normalizado
  - [x] `test_chunk_em_sentencas` — split correto, filtro de tamanho

### 1.5 Executar pré-processamento completo

- [x] Rodar `python scripts/preprocess.py` sobre as 5.960 bulas
- [x] Verificar `data/chunks_bulas.jsonl` — esperado: ~30.000 a 50.000 chunks
- [x] Verificar log: quantos arquivos sem seção de interação (esperado < 10%)

---

## Artefatos Produzidos

```
scripts/preprocess.py
tests/test_preprocess.py
data/chunks_bulas.jsonl (~30-50k linhas)
```

---

## Verificação

- [x] `python -m pytest tests/test_preprocess.py -v` — todos passam
- [x] `wc -l data/chunks_bulas.jsonl` — > 10.000 chunks
- [x] `head -3 data/chunks_bulas.jsonl | python -m json.tool` — JSON válido
- [x] Commit: `git add scripts/preprocess.py tests/test_preprocess.py && git commit -m "feat: Fase 1 — pré-processamento das bulas — 8 testes"`
