# Guia de Validacao — Anotacoes de Interacoes Medicamentosas

**Objetivo:** Ensinar outro agente (ou revisor humano) a validar a qualidade
dos pares anotados gerados pelo `scripts/annotate.py`, garantindo que o dataset
de fine-tuning tenha alta precisao.

**Arquivos a validar:**
- `data/anotacoes/automaticas.csv` — ~695 pares, Fonte 2, heuristica
- `data/anotacoes/pendentes_curadoria.csv` — 500 candidatos Fonte 1, para revisao
- `data/anotacoes/train.csv`, `val.csv`, `test.csv` — datasets balanceados (gerados apos validacao)

---

## 1. Carga Rapida dos Dados

```python
import pandas as pd

# Carregar automaticas
auto = pd.read_csv("data/anotacoes/automaticas.csv")
print(f"Automaticas: {len(auto)} pares")
print(auto["classe"].value_counts().to_dict())
# Esperado: {1: ~580, 2: ~60, 0: ~50}

# Carregar pendentes de curadoria
pend = pd.read_csv("data/anotacoes/pendentes_curadoria.csv")
print(f"Pendentes: {len(pend)} candidatos")
```

---

## 2. Regras de Ouro para Validacao

### Regra 1: O contexto DEVE mencionar o medicamento_outro

Um par so e valido se o `medicamento_outro` aparece **explicitamente** no campo `contexto`.
Se o contexto fala de interacoes "com anticoagulantes" mas nao cita "varfarina"
especificamente, o par **nao e valido** — remova.

```python
def regra1_medicamento_presente(row):
    """Verifica que medicamento_outro aparece no contexto."""
    outro = str(row["medicamento_outro"]).lower()
    contexto = str(row["contexto"]).lower()
    return outro in contexto
```

### Regra 2: A classe deve corresponder a gravidade do texto

Use a tabela do GUIA_ANOTACAO.md:

| Classe | Palavras-chave (qualquer uma) |
|---|---|
| 0 (SEM) | "nao ha interacao", "sem interacao", "nao foram observadas", "seguro", "pode ser usado" |
| 1 (LEVE) | "monitorar", "ajustar", "cautela", "precaucao", "pode aumentar/reduzir" |
| 2 (GRAVE) | "contraindicado", "fatal", "risco de morte", "nao administrar", "rabdomiolise" |

**Criterio:** Se o texto contem palavras de classe 2, a classe DEVE ser 2.
Palavras de classe 2 tem precedencia sobre classe 1.

```python
GRAVE = ["contraindicado", "fatal", "risco de morte", "nao administrar",
         "rabdomiolise", "stevens-johnson", "hemorragia grave"]
SEM = ["nao ha interacao", "sem interacao", "nao foram observadas",
       "nao apresenta interacao", "nenhuma interacao", "e seguro"]

def regra2_gravidade(row):
    """Verifica consistencia entre texto e classe."""
    texto = str(row["contexto"]).lower()
    classe = row["classe"]

    # Palavras GRAVE implicam classe 2
    for kw in GRAVE:
        if kw in texto and classe != 2:
            return False, f"Contem '{kw}' mas classe={classe} (deveria ser 2)"

    # Palavras SEM implicam classe 0
    for kw in SEM:
        if kw in texto and classe != 0:
            return False, f"Contem '{kw}' mas classe={classe} (deveria ser 0)"

    return True, "OK"
```

### Regra 3: O medicamento_alvo nao pode ser igual ao medicamento_outro

Self-interactions are meaningless.

```python
def regra3_sem_self(row):
    return str(row["medicamento_alvo"]).lower() != str(row["medicamento_outro"]).lower()
```

### Regra 4: Contexto deve ter tamanho minimo

Contextos menores que 30 caracteres provavelmente sao ruido.

```python
def regra4_tamanho_minimo(row):
    return len(str(row["contexto"])) >= 30
```

---

## 3. Script de Validacao Automatica

O agente validador deve executar este script e revisar manualmente os pares
que falham em qualquer regra:

```python
"""
Script de validacao de anotacoes.
Executa regras automaticas e gera relatorio de pares suspeitos para revisao manual.

Uso: python scripts/validate_annotations.py
"""

import pandas as pd
from pathlib import Path

# ─── Carregar ───
ANOTACOES_DIR = Path("data/anotacoes")
auto = pd.read_csv(ANOTACOES_DIR / "automaticas.csv")

print(f"Total de pares: {len(auto)}")
print(f"Distribuicao: {auto['classe'].value_counts().to_dict()}")

# ─── Regra 1: medicamento_outro presente no contexto ───
def check_medicamento_presente(row):
    outro = str(row["medicamento_outro"]).lower().strip()
    ctx = str(row["contexto"]).lower()
    return outro in ctx

auto["r1_ok"] = auto.apply(check_medicamento_presente, axis=1)
r1_fail = auto[~auto["r1_ok"]]
print(f"\nRegra 1 (medicamento no contexto): {len(r1_fail)} falhas")
if len(r1_fail) > 0:
    print("  Exemplos:")
    for _, r in r1_fail.head(3).iterrows():
        print(f"    alvo={r['medicamento_alvo']} outro={r['medicamento_outro']}")

# ─── Regra 2: consistencia classe vs gravidade ───
GRAVE_KW = ["contraindicado", "fatal", "risco de morte", "nao administrar",
            "rabdomiolise", "stevens-johnson"]
SEM_KW = ["nao ha interacao", "sem interacao", "nao foram observadas",
          "nenhuma interacao", "e seguro"]

def check_gravidade(row):
    texto = str(row["contexto"]).lower()
    classe = row["classe"]
    for kw in GRAVE_KW:
        if kw in texto and classe != 2:
            return False
    for kw in SEM_KW:
        if kw in texto and classe != 0:
            return False
    return True

auto["r2_ok"] = auto.apply(check_gravidade, axis=1)
r2_fail = auto[~auto["r2_ok"]]
print(f"Regra 2 (gravidade consistente): {len(r2_fail)} falhas")

# ─── Regra 3: sem self-interaction ───
auto["r3_ok"] = auto.apply(
    lambda r: str(r["medicamento_alvo"]).lower() != str(r["medicamento_outro"]).lower(),
    axis=1
)
r3_fail = auto[~auto["r3_ok"]]
print(f"Regra 3 (sem self): {len(r3_fail)} falhas")

# ─── Regra 4: tamanho minimo do contexto ───
auto["r4_ok"] = auto["contexto"].apply(lambda x: len(str(x)) >= 30)
r4_fail = auto[~auto["r4_ok"]]
print(f"Regra 4 (tamanho >= 30): {len(r4_fail)} falhas")

# ─── Consolida falhas para revisao ───
auto["falhou"] = ~(auto["r1_ok"] & auto["r2_ok"] & auto["r3_ok"] & auto["r4_ok"])
suspicious = auto[auto["falhou"]]
print(f"\n=== RESUMO ===")
print(f"Pares OK: {len(auto) - len(suspicious)}")
print(f"Pares SUSPEITOS: {len(suspicious)} ({100*len(suspicious)/len(auto):.1f}%)")
print(f"  Para revisao manual: data/anotacoes/suspicious_review.csv")

# Salva suspeitos para revisao
suspicious.to_csv(ANOTACOES_DIR / "suspicious_review.csv", index=False)

# ─── Metricas de qualidade ───
print(f"\n=== ESTATISTICAS ===")
print(f"Distribuicao original:  {auto['classe'].value_counts().to_dict()}")
print(f"Distribuicao apos filtro: {auto[~auto['falhou']]['classe'].value_counts().to_dict()}")
print(f"Taxa de rejeicao: {len(suspicious)}/{len(auto)} = {100*len(suspicious)/len(auto):.1f}%")
```

---

## 4. Revisao Manual dos Casos Suspeitos

Apos rodar o script de validacao, o agente deve revisar cada linha de
`data/anotacoes/suspicious_review.csv` e decidir:

### Acao 1: CORRIGIR (ajustar classe)

Se o erro e apenas de classificacao:
- O texto diz "contraindicado" mas a classe e 1 → mudar para **classe 2**
- O texto diz "nao ha interacao" mas a classe e 1 → mudar para **classe 0**

### Acao 2: REMOVER (par invalido)

Se o par nao faz sentido:
- `medicamento_outro` nao aparece no contexto → **remover**
- `medicamento_alvo == medicamento_outro` → **remover**
- Contexto nao fala de interacao medicamentosa (ex: fala de alimento, exame laboratorial, condicao clinica) → **remover**

### Acao 3: MANTER (falso positivo das regras)

Se a regra falhou mas a anotacao esta correta:
- Ex: o contexto menciona "anticoagulantes orais" e o medicamento_outro e "varfarina" —
  a regra 1 falha (string "varfarina" nao esta no texto), mas a associacao e clinicamente correta → **manter**

### Como marcar a decisao

Adicione uma coluna `decisao` ao CSV de suspeitos:

```python
suspicious = pd.read_csv("data/anotacoes/suspicious_review.csv")
suspicious["decisao"] = ""  # CORRIGIR_CLASSE_X, REMOVER, ou MANTER
suspicious["nova_classe"] = -1  # preencher apenas se CORRIGIR
suspicious.to_csv("data/anotacoes/suspicious_reviewed.csv", index=False)
```

---

## 5. Reconstrucao do Dataset Limpo

Apos revisar todos os suspeitos, gere o dataset final:

```python
import pandas as pd

auto = pd.read_csv("data/anotacoes/automaticas.csv")
review = pd.read_csv("data/anotacoes/suspicious_reviewed.csv")

# Aplica correcoes
for _, r in review.iterrows():
    idx = r.name  # indice original
    if r["decisao"] == "REMOVER":
        auto.drop(idx, inplace=True)
    elif r["decisao"].startswith("CORRIGIR"):
        nova = int(r["decisao"].split("_")[-1])
        auto.at[idx, "classe"] = nova
    # MANTER → nao faz nada

print(f"Pares apos limpeza: {len(auto)}")
print(auto["classe"].value_counts().to_dict())

auto.to_csv("data/anotacoes/automaticas_limpas.csv", index=False)
```

---

## 6. Validacao da Curadoria Manual (Fonte 1)

O arquivo `pendentes_curadoria.csv` contem 500 candidatos da Fonte 1
para anotacao humana. Cada linha tem:

| Campo | Descricao |
|---|---|
| `medicamento` | Medicamento alvo da bula |
| `secao` | Secao da bula |
| `texto` | Contexto (chunk) |
| `classe_sugerida` | Classe sugerida pela heuristica (-1 = nao classificado) |
| `confianca_heuristica` | Confianca da heuristica (0-1, -1 se nao classificado) |
| `medicamentos_detectados` | Medicamentos encontrados via lista de frequentes |
| `classe_manual` | **VAZIO** — preencher com 0, 1 ou 2 |

### Como preencher classe_manual

1. Leia o `texto`
2. Identifique o `medicamento` alvo (nome da bula)
3. Identifique os medicamentos mencionados (use `medicamentos_detectados` como guia)
4. Para CADA medicamento_outro encontrado:
   - Classifique conforme as regras do GUIA_ANOTACAO.md
   - Crie um novo par no CSV `manuais.csv`

### Exemplo

Linha do `pendentes_curadoria.csv`:
```
medicamento: amoxicilina
texto: "A probenecida reduz a secrecao tubular renal da amoxicilina..."
medicamentos_detectados: probenecida
classe_sugerida: 1
```

Voce cria em `manuais.csv`:
```csv
medicamento_alvo,medicamento_outro,contexto,classe,fonte,origem,confianca
amoxicilina,probenecida,"A probenecida reduz a secrecao tubular renal...",1,fonte1,manual,1.0
```

---

## 7. Checklist Final de Qualidade

Antes de considerar o dataset pronto para fine-tuning:

- [ ] **Total de pares:** ≥ 1.200 (automaticos limpos + manuais)
- [ ] **Distribuicao:** classe 0 ≥ 20%, classe 1 ≥ 30%, classe 2 ≥ 10%
- [ ] **Regra 1 (medicamento presente):** ≥ 95% passam
- [ ] **Regra 2 (gravidade consistente):** ≥ 90% passam
- [ ] **Regra 3 (sem self):** 100% passam
- [ ] **Regra 4 (tamanho minimo):** ≥ 98% passam
- [ ] **Nomes normalizados:** lowercase, sem acentos
- [ ] **Sem duplicatas:** `(alvo, outro, contexto[:100])` unico
- [ ] **CSV carrega sem erro:** `pd.read_csv()` funciona
- [ ] **Coluna classe:** apenas 0, 1, 2
- [ ] **Coluna confianca:** 0.0 a 1.0

---

## 8. Exemplo de Sessao Completa de Validacao

```bash
# 1. Gerar automaticas (Fonte 2)
python scripts/annotate.py --chunks data/chunks_bulas.jsonl --output data/anotacoes

# 2. Validar automaticamente
python scripts/validate_annotations.py

# 3. Revisar suspeitos manualmente
#    → Abrir data/anotacoes/suspicious_review.csv
#    → Adicionar coluna 'decisao' (CORRIGIR_CLASSE_0, REMOVER, MANTER)
#    → Salvar como suspicious_reviewed.csv

# 4. Reconstruir dataset limpo
python scripts/rebuild_clean_dataset.py

# 5. Verificar metricas finais
python scripts/validate_annotations.py --input data/anotacoes/automaticas_limpas.csv
```
