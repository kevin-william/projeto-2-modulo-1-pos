"""
Script de validacao automatica de anotacoes.
Executa regras e gera relatorio de pares suspeitos para revisao manual.

Uso: python scripts/validate_annotations.py
"""

import pandas as pd
from pathlib import Path

ANOTACOES_DIR = Path("data/anotacoes")

# ─── Carregar ───
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
    axis=1,
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

if len(suspicious) > 0:
    suspicious.to_csv(ANOTACOES_DIR / "suspicious_review.csv", index=False)
    print(f"  Para revisao manual: data/anotacoes/suspicious_review.csv")

# ─── Metricas de qualidade ───
print(f"\n=== ESTATISTICAS ===")
print(f"Distribuicao original:     {auto['classe'].value_counts().to_dict()}")
print(f"Distribuicao apos filtro:  {auto[~auto['falhou']]['classe'].value_counts().to_dict()}")
print(f"Taxa de rejeicao: {len(suspicious)}/{len(auto)} = {100*len(suspicious)/len(auto):.1f}%")
