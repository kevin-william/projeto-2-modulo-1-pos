# Fase 3: Anotação do Dataset de Treino

**Objetivo:** Criar ~1.500 pares rotulados (medicamento_alvo, medicamento_outro, contexto, classe) usando weak supervision da Fonte 2 + curadoria manual, para fine-tuning do BioBERTpt.

**Dependências:** Fase 1 concluída (chunks disponíveis).

---

## Estratégia de Anotação

### Passo A: Weak supervision automática (~1.000 pares)

Usar a Fonte 2 (Consultaremedios) como "supervisão distante". O bloco `INTERAÇÃO MEDICAMENTOSA?` contém frases conclusivas que permitem classificação heurística:

- **Classe 2 (GRAVE):** "contraindicado", "fatal", "risco grave", "nunca associar", "não administrar", "risco de morte"
- **Classe 1 (LEVE_MODERADA):** "monitorar", "ajustar dose", "precaução", "potencializa", "diminui efeito", "cautela", "recomenda-se"
- **Classe 0 (SEM_INTERACAO):** "não há interação", "nenhuma interação", "seguro", "pode ser usado", "não foram observadas"

### Passo B: Curadoria manual (~500 pares)

Selecionar casos ambíguos da Fonte 1 que a heurística não conseguiu classificar com confiança, e anotar manualmente.

---

## Tarefas

### 3.1 Criar `scripts/annotate.py`

- [ ] Função `classificar_heuristicamente(texto: str) -> tuple[int, float]`
  - Busca por padrões regex para cada classe
  - Retorna `(classe, confiança)` onde confiança é baseada na força do match
  - Se múltiplos matches: prioriza classe mais grave com maior confiança
  - Se nenhum match: retorna `(-1, -1.0)` (não classificado)

- [ ] Função `gerar_pares_automaticos(chunks: list[dict]) -> list[dict]`
  - Para cada chunk da Fonte 2:
    1. Executar NER (`clinicalnerpt-chemical`) para extrair entidades
    2. Extrair medicamento-alvo (do metadata do chunk)
    3. Para cada outra entidade no mesmo chunk:
       - Criar par `(alvo, outro, texto_chunk)`
       - Classificar heuristicamente
       - Se confiança ≥ 0.7: incluir no dataset automático
  - Alvo: ~1.000 pares

- [ ] Função `selecionar_para_curadoria(chunks: list[dict], n: int = 500) -> list[dict]`
  - Priorizar chunks da Fonte 1 (mais ambíguos)
  - Priorizar chunks com múltiplas entidades (mais complexos)
  - Priorizar chunks onde heurística teve baixa confiança
  - Garantir balanceamento aproximado entre classes

- [ ] Função `exportar_csv(pares: list[dict], caminho: Path)`
  - Colunas: `medicamento_alvo`, `medicamento_outro`, `contexto`, `classe`, `confianca`, `fonte`, `origem`
  - `origem`: `"automatica"` ou `"manual"`

### 3.2 Executar weak supervision

- [ ] Rodar `gerar_pares_automaticos()` sobre todos os chunks da Fonte 2
- [ ] Salvar `data/anotacoes/automaticas.csv`
- [ ] Verificar: ≥ 800 pares, distribuição razoável entre classes

### 3.3 Preparar lotes para curadoria manual

- [ ] Rodar `selecionar_para_curadoria(n=500)`
- [ ] Salvar `data/anotacoes/pendentes_curadoria.csv` (sem classe preenchida)
- [ ] Formatar para fácil anotação: cada linha com contexto + medicamentos, campo `classe` vazio

### 3.4 Realizar curadoria manual

- [ ] Anotar os ~500 pares manualmente (fora do script — trabalho humano)
- [ ] Preencher classe (0, 1, ou 2) para cada par
- [ ] Salvar como `data/anotacoes/manuais.csv`
- [ ] Revisar: consistência entre anotações, casos borderline

### 3.5 Consolidar dataset final

- [ ] Concatenar `automaticas.csv` + `manuais.csv`
- [ ] Balancear: undersample classe majoritária se necessário
- [ ] Split: treino (80%), validação (10%), teste (10%)
- [ ] Salvar:
  - `data/anotacoes/train.csv`
  - `data/anotacoes/val.csv`
  - `data/anotacoes/test.csv`
- [ ] Estatísticas: total de pares, distribuição por classe, média de tokens por contexto

### 3.6 Testes

- [ ] `tests/test_annotate.py`:
  - [ ] `test_classificar_heuristicamente_grave` — frase com "contraindicado" → (2, >0.7)
  - [ ] `test_classificar_heuristicamente_leve` — frase com "monitorar" → (1, >0.5)
  - [ ] `test_classificar_heuristicamente_sem` — frase com "não há interação" → (0, >0.8)
  - [ ] `test_classificar_heuristicamente_ambiguo` — frase sem padrões → (-1, -1.0)
  - [ ] `test_classificar_heuristicamente_multiplos` — múltiplos padrões → prioriza mais grave
  - [ ] `test_gerar_pares_automaticos` — com 5 chunks de amostra → ≥ 3 pares gerados
  - [ ] `test_exportar_csv` — arquivo CSV gerado com colunas corretas

---

## Artefatos Produzidos

```
scripts/annotate.py
tests/test_annotate.py
data/anotacoes/automaticas.csv     (~1.000 pares)
data/anotacoes/pendentes_curadoria.csv
data/anotacoes/manuais.csv         (~500 pares)
data/anotacoes/train.csv           (80%)
data/anotacoes/val.csv             (10%)
data/anotacoes/test.csv            (10%)
```

---

## Verificação

- [ ] `python -m pytest tests/test_annotate.py -v` — todos passam
- [ ] `wc -l data/anotacoes/train.csv` — ≥ 1.000 linhas (header + dados)
- [ ] Distribuição: classe 0 ≥ 30%, classe 1 ≥ 25%, classe 2 ≥ 15% do total
- [ ] `python scripts/annotate.py --stats` exibe estatísticas de平衡
- [ ] Commit: `git add scripts/annotate.py tests/test_annotate.py data/anotacoes/*.csv && git commit -m "feat: Fase 3 — anotação do dataset — 7 testes"`
