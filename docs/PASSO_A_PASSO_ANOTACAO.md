# Guia Passo a Passo — Anotação Manual (Fonte 1)

**Objetivo:** Transformar os 500 candidatos de `pendentes_curadoria.csv` em
pares rotulados no arquivo `manuais.csv`, seguindo as regras do
`GUIA_ANOTACAO.md`.

**Quem vai usar este guia:** Uma pessoa (ou agente) que **nunca viu o projeto**.
Tudo está explicado do zero. Não é preciso conhecer Python, machine learning
ou o domínio farmacêutico — apenas ler com atenção e seguir os passos.

**Tempo estimado:** 2 a 4 horas (30 segundos a 1 minuto por candidato).

---

## 0. O que você precisa ter em mãos

Apenas estes dois arquivos (já existem no projeto):

| Arquivo | O que contém |
|---|---|
| `data/anotacoes/pendentes_curadoria.csv` | 500 textos de bulas + metadados. É a sua **matéria-prima**. |
| `docs/GUIA_ANOTACAO.md` | Regras de classificação. É o seu **manual de consulta**. |

Abra os dois. Você vai alternar entre eles.

---

## 1. Entendendo o arquivo de entrada

`pendentes_curadoria.csv` pode ser aberto no **Excel, Google Sheets ou LibreOffice Calc**.

Cada linha representa **um trecho de bula** que precisa ser analisado. As colunas são:

| Coluna | Significado | Exemplo |
|---|---|---|
| `medicamento` | Nome do remédio "dono" da bula | `amoxicilina` |
| `secao` | Seção da bula de onde o texto veio | `INTERAÇÕES MEDICAMENTOSAS` |
| `texto` | O trecho da bula que você vai ler | `"A probenecida reduz a secrecao tubular renal..."` |
| `classe_sugerida` | Palpite do robô (-1 = "não sei") | `1` ou `-1` |
| `confianca_heuristica` | Quão confiante o robô estava (0 a 1) | `0.75` |
| `medicamentos_detectados` | Remédios que o robô encontrou no texto | `probenecida; alopurinol` |
| `score_prioridade` | Nota de 0 a 10 (quanto maior, mais importante) | `8.5` |
| `classe_manual` | **VAZIO** — é aqui que VOCÊ escreve | _(em branco)_ |

---

## 2. As 3 classes possíveis

Você só pode escrever **0**, **1** ou **2** na coluna `classe_manual`. O significado:

| Classe | Nome | Significado |
|---|---|---|
| **0** | SEM INTERAÇÃO | A bula diz que **não há** interação ou que é **seguro** usar junto |
| **1** | LEVE / MODERADA | Há interação, mas o médico só precisa **monitorar** ou **ajustar a dose** |
| **2** | GRAVE / CONTRAINDICADA | A bula diz que é **contraindicado**, que há **risco de morte** ou evento **grave** |

---

## 3. Como decidir a classe (árvore de decisão)

Para cada linha, leia a coluna `texto` e siga este fluxo:

```
PERGUNTA 1: O texto fala de outro remédio além do "medicamento" da bula?
   ├── NÃO → PULE esta linha (não anote nada, deixe classe_manual em branco)
   └── SIM → Continue

PERGUNTA 2: O texto diz qual o EFEITO de usar os dois juntos?
   ├── NÃO (apenas cita o nome, sem dizer o que acontece) → PULE
   └── SIM → Continue

PERGUNTA 3: O texto usa palavras de ALERTA GRAVE?
   Palavras como: "contraindicado", "risco de morte", "fatal",
   "não deve ser administrado", "nunca associar", "rabdomiólise",
   "hemorragia grave", "Stevens-Johnson"
   ├── SIM → Classe 2
   └── NÃO → Continue

PERGUNTA 4: O texto diz que NÃO há interação?
   Frases como: "não há interação", "não foram observadas interações",
   "sem interação medicamentosa", "é seguro", "pode ser usado"
   ├── SIM → Classe 0
   └── NÃO → Continue

PERGUNTA 5: O texto descreve alguma interação (mesmo que leve)?
   Palavras como: "monitorar", "ajustar dose", "cautela", "precaução",
   "pode aumentar", "pode reduzir", "recomenda-se"
   ├── SIM → Classe 1
   └── NÃO → PULE (caso raro — texto ambíguo)
```

---

## 4. Exemplos práticos

### Exemplo 1 — Classe 1 (LEVE)

**Texto:** `"A probenecida reduz a secrecao tubular renal da amoxicilina.
No uso concomitante com amoxicilina, pode haver aumento dos niveis
de amoxicilina no sangue."`

**Análise:**
- Medicamento da bula: `amoxicilina`
- Outro remédio mencionado: `probenecida` ✓
- Descreve efeito? Sim — "aumento dos níveis" ✓
- Tem palavra grave? Não ✗
- Diz que não há interação? Não ✗
- Descreve interação? Sim — "reduz a secreção", "pode haver aumento" ✓

**Decisão:** Classe **1**

---

### Exemplo 2 — Classe 2 (GRAVE)

**Texto:** `"O uso concomitante de Amoxicilina com Metotrexato e
contraindicado devido ao risco de toxicidade grave e potencialmente fatal."`

**Análise:**
- Medicamento da bula: `amoxicilina`
- Outro remédio: `metotrexato` ✓
- Descreve efeito? Sim ✓
- Tem palavra grave? **SIM** — "contraindicado", "fatal" ✓

**Decisão:** Classe **2** (nem precisa continuar)

---

### Exemplo 3 — Classe 0 (SEM INTERAÇÃO)

**Texto:** `"Não há interações clinicamente relevantes com paracetamol
quando utilizado nas doses recomendadas."`

**Análise:**
- Medicamento da bula: `atorvastatina`
- Outro remédio: `paracetamol` ✓
- Tem palavra grave? Não ✗
- Diz que NÃO há interação? **SIM** — "não há interações" ✓

**Decisão:** Classe **0**

---

### Exemplo 4 — PULAR (não anotar)

**Texto:** `"Recomenda-se que testes de função hepática sejam realizados
antes do início do tratamento e periodicamente durante o mesmo."`

**Análise:**
- Medicamento da bula: `atorvastatina`
- Outro remédio mencionado? **NÃO** — não cita nenhum outro medicamento ✗

**Decisão:** **PULAR** — não preencher `classe_manual`

---

### Exemplo 5 — PULAR (apenas cita, sem descrever efeito)

**Texto:** `"São conhecidas outras interações medicamentosas. Avise seu
médico se você fizer uso de antiácidos, colestipol, contraceptivos orais,
varfarina, ácido fusídico."`

**Análise:**
- Outros remédios: `antiácidos`, `colestipol`, `contraceptivos orais`,
  `varfarina`, `ácido fusídico` ✓
- Descreve o efeito de cada um? **NÃO** — apenas lista nomes, sem dizer
  o que acontece com cada um ✗

**Decisão:** **PULAR** — informação insuficiente para classificar

---

## 5. Passo a passo prático (o que fazer de fato)

### 5.1 Abrir o arquivo

1. Abra o Excel (ou Google Sheets)
2. Arquivo → Abrir → navegue até `C:\workspace\python\projeto-2-modulo-1-pos\data\anotacoes\`
3. Selecione `pendentes_curadoria.csv`
4. Se asked about delimiter, escolha **vírgula** (`,`)
5. Aumente a largura da coluna `texto` para conseguir ler

### 5.2 Organizar a tela

Deixe duas janelas abertas lado a lado:
- **Esquerda:** `pendentes_curadoria.csv` (Excel)
- **Direita:** este guia (`PASSO_A_PASSO_ANOTACAO.md`) ou `GUIA_ANOTACAO.md`

### 5.3 Começar pelas mais importantes

Ordene pela coluna `score_prioridade` (decrescente). As linhas com score
mais alto são as mais promissoras — têm mais medicamentos detectados e
a heurística do robô ficou mais confusa (são os casos que mais precisam
de um humano).

No Excel: clique na coluna `score_prioridade` → Dados → Ordenar do maior para o menor.

### 5.4 Para cada linha

1. **Leia** a coluna `texto`
2. **Identifique** qual é o medicamento da bula (coluna `medicamento`)
3. **Identifique** quais outros remédios aparecem no texto (use
   `medicamentos_detectados` como dica, mas confira — às vezes o robô erra)
4. **Aplique a árvore de decisão** da seção 3
5. **Escreva** 0, 1 ou 2 na coluna `classe_manual`
6. Se não souber ou o texto for ambíguo, **deixe em branco** e passe para a próxima

### 5.5 Meta diária

- **Sessão 1 (1 hora):** 150-200 linhas
- **Sessão 2 (1 hora):** 150-200 linhas
- **Sessão 3 (30 min):** Revisar as que ficaram em branco

---

## 6. Casos especiais

### 6.1 Texto menciona vários remédios

Se o texto fala de 3 remédios diferentes (ex: probenecida, alopurinol e
varfarina) e descreve o efeito de cada um:

**Você precisa criar linhas separadas no `manuais.csv`** — uma para cada par.

Exemplo:
```
Texto: "A probenecida reduz a secrecao da amoxicilina. O alopurinol pode
aumentar reacoes alergicas. A varfarina requer monitoramento do INR."

→ 3 pares:
  1. amoxicilina + probenecida → classe 1
  2. amoxicilina + alopurinol → classe 1
  3. amoxicilina + varfarina → classe 1
```

Mas isso será feito na etapa 7 (geração do CSV), não na marcação do
`classe_manual`. Por enquanto, na coluna `classe_manual`, coloque a classe
da interação **mais grave** encontrada no texto.

### 6.2 Texto muito técnico que você não entende

Se o texto usar termos muito técnicos e você não tiver certeza:

1. Tente identificar as palavras-chave (seção 3)
2. Se ainda assim não souber → **deixe em branco**
3. É melhor ter menos anotações corretas do que muitas incorretas

### 6.3 O robô sugere classe -1 (não classificado)

Isso significa que a heurística não encontrou palavras-chave no texto.
**Não confie cegamente.** Leia o texto você mesmo — muitas vezes o robô
perde interações descritas com palavras diferentes.

---

## 7. Gerando o arquivo final `manuais.csv`

Após preencher a coluna `classe_manual` para todos os candidatos relevantes,
você precisa gerar o arquivo `manuais.csv` que contém os **pares
medicamentosos individuais**.

### 7.1 Formato do `manuais.csv`

Cada linha representa **UM par** de medicamentos:

```csv
medicamento_alvo,medicamento_outro,contexto,classe,fonte,origem,confianca
amoxicilina,probenecida,"A probenecida reduz a secrecao tubular renal da amoxicilina...",1,fonte1,manual,1.0
amoxicilina,varfarina,"Existem casos raros de INR aumentada em pacientes mantidos com varfarina...",1,fonte1,manual,1.0
captopril,ibuprofeno,"Os AINEs como ibuprofeno podem reduzir o efeito anti-hipertensivo do captopril...",1,fonte1,manual,1.0
```

### 7.2 Como preencher cada coluna

| Coluna | Como preencher |
|---|---|
| `medicamento_alvo` | Copie da coluna `medicamento` do `pendentes_curadoria.csv`. **Sempre em minúsculas, sem acentos.** |
| `medicamento_outro` | O outro remédio mencionado no texto. Use a coluna `medicamentos_detectados` como referência. Se houver vários, crie uma linha para cada um. |
| `contexto` | Copie a coluna `texto`. **Entre aspas duplas** se contiver vírgulas. |
| `classe` | Copie da coluna `classe_manual` (0, 1 ou 2). |
| `fonte` | Sempre `fonte1` |
| `origem` | Sempre `manual` |
| `confianca` | `1.0` se você tem certeza. `0.7` se ficou em dúvida. |

### 7.3 Como criar o arquivo (3 opções)

**Opção A — No próprio Excel (recomendado para iniciantes):**

1. Termine de preencher `classe_manual` no `pendentes_curadoria.csv`
2. Salve e feche
3. Abra um novo arquivo Excel em branco
4. Crie as 7 colunas: `medicamento_alvo`, `medicamento_outro`, `contexto`,
   `classe`, `fonte`, `origem`, `confianca`
5. Para cada linha do `pendentes_curadoria.csv` que tem `classe_manual`
   preenchido:
   - Identifique os medicamentos mencionados (use `medicamentos_detectados`)
   - Para CADA medicamento, crie uma nova linha no seu arquivo
6. Salve como `manuais.csv` (CSV, separado por vírgula)

**Opção B — Com script Python (se souber programar):**

```python
import pandas as pd
from pathlib import Path

# Carregar
pend = pd.read_csv("data/anotacoes/pendentes_curadoria.csv")
# Filtrar apenas os que foram anotados
anotados = pend[pend["classe_manual"].notna() & (pend["classe_manual"] >= 0)]

pares = []
for _, row in anotados.iterrows():
    # Extrair medicamentos do campo medicamentos_detectados
    outros_str = str(row["medicamentos_detectados"])
    outros = [m.strip() for m in outros_str.split(";") if m.strip()]

    for outro in outros:
        pares.append({
            "medicamento_alvo": str(row["medicamento"]).lower().strip(),
            "medicamento_outro": outro.lower().strip(),
            "contexto": str(row["texto"]),
            "classe": int(row["classe_manual"]),
            "fonte": "fonte1",
            "origem": "manual",
            "confianca": 1.0,
        })

manuais = pd.DataFrame(pares)
manuais.to_csv("data/anotacoes/manuais.csv", index=False)
print(f"manuais.csv gerado com {len(manuais)} pares")
```

**Opção C — Manualmente no Bloco de Notas:**

1. Abra o Bloco de Notas
2. Escreva a primeira linha: `medicamento_alvo,medicamento_outro,contexto,classe,fonte,origem,confianca`
3. Para cada par, escreva uma linha no formato CSV
4. Salve como `data/anotacoes/manuais.csv`

---

## 8. Checklist de verificação

Antes de considerar o trabalho concluído:

- [ ] `pendentes_curadoria.csv` tem a coluna `classe_manual` preenchida para
  pelo menos 200 linhas (ideal: 300+)
- [ ] `manuais.csv` existe e tem pelo menos 300 linhas (pares)
- [ ] `manuais.csv` abre no Excel sem erro
- [ ] Nenhuma linha de `manuais.csv` tem `classe` vazio
- [ ] Todos os valores de `classe` são 0, 1 ou 2 (nada de 3, -1, "grave", etc.)
- [ ] Todos os valores de `fonte` são `fonte1`
- [ ] Todos os valores de `origem` são `manual`
- [ ] Nomes de medicamentos estão em **minúsculas** e **sem acentos**
- [ ] Nenhum par tem `medicamento_alvo` igual a `medicamento_outro`

---

## 9. Dúvidas comuns

**P: E se o texto mencionar um remédio, mas eu não souber escrever o nome certo?**
R: Use exatamente o nome que aparece na coluna `medicamentos_detectados`.
O robô já normalizou (minúsculas, sem acentos). Se o nome não estiver lá,
escreva como aparece no texto, mas em minúsculas e sem acentos.

**P: O que fazer com textos que falam de "classe de medicamentos" (ex: "anticoagulantes", "AINEs") sem nomear um específico?**
R: Se o texto só fala da classe e não cita nenhum medicamento específico
(ex: "varfarina", "ibuprofeno"), **não anote**. "Anticoagulantes" não é
um medicamento, é uma categoria.

**P: Posso pular os textos mais difíceis e anotar só os fáceis?**
R: Sim. É melhor ter 200 anotações corretas do que 500 com erro.
Priorize os textos onde você tem certeza da classe.

**P: Depois que eu gerar o `manuais.csv`, o que faço com ele?**
R: Entregue o arquivo para a pessoa responsável pelo projeto (Kevin).
O arquivo será combinado com `automaticas.csv` para formar o dataset
de fine-tuning.
