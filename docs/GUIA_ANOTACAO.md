# Guia de Anotação — Dataset de Interações Medicamentosas

**Objetivo:** Criar pares rotulados `(medicamento_alvo, medicamento_outro, contexto, classe)` para fine-tuning do BioBERTpt, extraídos das bulas pré-processadas.

**Projeto de referência:** `C:\workspace\python\python-processador-bulas` — contém as bulas já podadas (seções de interação) e o JSONL com metadados.  
**Projeto destino:** `C:\workspace\python\projeto-2-modulo-1-pos` — onde o dataset anotado será salvo.

---

## 1. Estrutura dos Dados de Referência

```
C:\workspace\python\python-processador-bulas\
├── data\
│   ├── pruned\
│   │   ├── fonte1\              ← 4.978 bulas ANVISA podadas
│   │   │   ├── 105830895_amoxicilina_profissional.txt
│   │   │   ├── 105830895_amoxicilina_paciente.txt
│   │   │   ├── 100380098_captopril_profissional.txt
│   │   │   └── ... (uma bula = um .txt, seções clínicas apenas)
│   │   ├── fonte2\              ← 982 bulas Consultaremedios podadas
│   │   │   ├── zarator.txt
│   │   │   ├── zocor.txt
│   │   │   ├── zyloric.txt
│   │   │   └── ... (3 blocos Q&A por arquivo)
│   │   └── bulas_pruned.jsonl   ← 5.960 linhas, metadados de cada bula
│   └── metrics\
│       └── pruning_report.csv   ← Relatório de redução de tokens
```

### Schema do JSONL (uma linha)

```json
{
  "id": "105830895_amoxicilina_profissional",
  "fonte": "fonte1",
  "medicamento": "amoxicilina",
  "versao": "profissional",
  "fabricante": "GERMED FARMACÊUTICA LTDA",
  "texto_original_tokens": 15079,
  "texto_podado_tokens": 7699,
  "reducao_percentual": 48.9,
  "secoes_mantidas": ["CONTRAINDICAÇÕES", "ADVERTÊNCIAS E PRECAUÇÕES",
                       "INTERAÇÕES MEDICAMENTOSAS", "REAÇÕES ADVERSAS"],
  "blocos_mantidos": null,
  "status": "ok"
}
```

---

## 2. Formato dos Arquivos Podados

### Fonte 1 (ANVISA) — Texto corrido com seções demarcadas

Os arquivos da Fonte 1 contêm APENAS as seções clínicas relevantes. Cada seção é delimitada por um cabeçalho `## NOME_DA_SEÇÃO`. A seção mais importante para anotação é `## INTERAÇÕES MEDICAMENTOSAS`.

**Exemplo real** (`105830895_amoxicilina_profissional.txt`, trecho da seção de interações):

```
## INTERAÇÕES  MEDICAMENTOSAS

6. INTERAÇÕES  MEDICAMENTOSAS
A probenecida reduz a secreção tubular renal da amoxicilina. No uso concomitante com
amoxicilina, pode haver aumento dos níveis de amoxicilina no sangue e no prolongamento
dessa alteração.

Da mesma forma que outros antibióticos, amoxicilina pode afetar a flora intestinal,
levando a menor reabsorção de estrógenos, e reduzir a eficácia dos contraceptivos orais.

A administração concomitante de alopurinol durante o tratamento com amoxicilina pode
aumentar a probabilidade de reações alérgicas da pele.

Na literatura existem casos raros de INR aumentada em pacientes mantidos com acenocumarol
ou varfarina, ao receberem um curso de tratamento com amoxicilina. Se a coadministração é
necessária, o tempo de protrombina ou INR deve ser cuidadosamente monitorado, na introdução
e ao término do tratamento com amoxicilina.
```

**🔑 Como ler:**
- **Medicamento alvo**: extraído do nome do arquivo → `amoxicilina`
- **Medicamentos mencionados**: `probenecida`, `alopurinol`, `acenocumarol`, `varfarina`
- **Contexto**: cada parágrafo que menciona outro medicamento
- **Classificação**: depende da gravidade descrita

### Fonte 2 (Consultaremedios) — Blocos Q&A

Cada arquivo tem 3 blocos no formato `[P: PERGUNTA?]` + `R: resposta`. O bloco relevante é `[P: INTERAÇÃO MEDICAMENTOSA?]`.

**Exemplo real** (`zarator.txt`):

```
[P: INTERAÇÃO MEDICAMENTOSA?]
R: Interação medicamentosa: quais os efeitos de tomar Zarator com outros remédios?
[...]
Miopatia (dor ou fraqueza muscular) devido à lesão dos músculos pode ocorrer em
pacientes que usam Zarator, sendo mais frequentes naqueles que usam também
ciclosporina, fibratos, niacina ou antifúngicos azólicos.

A administração concomitante de Zarator com medicamentos inibidores do citocromo
P450 3A4 (por ex., ciclosporina, eritromicina/claritromicina, inibidores da protease,
cloridrato de diltiazem, suco de grapefruit) pode alterar a quantidade de
atorvastatina no sangue.

São conhecidas outras interações medicamentosas, avise seu médico se você fizer uso
de antiácidos, colestipol, contraceptivos orais (pílulas), varfarina, ácido fusídico.
```

**Exemplo real** (`zocor.txt`):

```
[P: INTERAÇÃO MEDICAMENTOSA?]
R: Interação medicamentosa: quais os efeitos de tomar Zocor com outros remédios?
[...]
É muito importante informar ao seu médico se você for tomar Zocor associado a
qualquer um dos medicamentos listados a seguir, pois o risco de problemas musculares
nessa situação é maior: agentes antifúngicos (como o itraconazol, cetoconazol,
posaconazol ou voriconazol); inibidores da protease do HIV (tais como indinavir,
nelfinavir, ritonavir e saquinavir); [...] ciclosporina; [...] amiodarona (medicamento
utilizado para o tratamento de arritmias cardíacas); verapamil, diltiazem [...]
```

**🔑 Como ler:**
- **Medicamento alvo**: extraído do nome do arquivo (ex: `zarator` → atorvastatina, `zocor` → sinvastatina)
- **Medicamentos mencionados**: listados explicitamente na resposta
- **Contexto**: toda a resposta do bloco `INTERAÇÃO MEDICAMENTOSA?`
- **Classificação**: baseada na gravidade descrita (ex: "risco de problemas musculares" pode ser classe 1 ou 2)

---

## 3. Regras de Classificação (3 Classes)

Use estas regras para decidir a classe de cada par `(medicamento_alvo, medicamento_outro)`.

### Classe 0 — SEM_INTERAÇÃO

**A bula afirma que NÃO há interação ou que é seguro.**

Palavras-chave e frases típicas:
- "não há interação"
- "não há interações conhecidas"
- "não foram observadas interações"
- "nenhuma interação clinicamente relevante"
- "pode ser administrado concomitantemente"
- "pode ser usado com segurança"
- "sem interação medicamentosa"
- "sem risco de interação"

**Exemplo real (Fonte 2, `zarator.txt`):**
> "Não há interações clinicamente relevantes com paracetamol quando utilizado nas doses recomendadas."
> → **Classe 0** (paracetamol + atorvastatina)

### Classe 1 — LEVE_MODERADA

**Há interação, mas o manejo é simples: monitoramento, ajuste de dose, ou precaução. NÃO há contraindicação absoluta.**

Palavras-chave e frases típicas:
- "monitorar", "monitoramento"
- "ajustar dose", "ajuste de dose"
- "usar com cautela", "recomenda-se cautela"
- "precaução"
- "potencializa efeito", "pode aumentar o efeito"
- "diminui a absorção", "reduz a eficácia"
- "pode interferir"
- "recomenda-se acompanhamento"
- "deve ser monitorado"
- "pode reduzir", "pode aumentar", "pode alterar"

**Exemplo real (Fonte 1, `105830895_amoxicilina_profissional.txt`):**
> "A probenecida reduz a secreção tubular renal da amoxicilina. No uso concomitante com amoxicilina, pode haver aumento dos níveis de amoxicilina no sangue."
> → **Classe 1** (amoxicilina + probenecida — interação farmacocinética, sem risco de vida)

> "Recomenda-se que, ao realizar testes para verificação da presença de glicose na urina durante o tratamento com amoxicilina, sejam usados métodos de glicose oxidase enzimática. Devido às altas concentrações urinárias de amoxicilina, leituras falso-positivas são comuns com métodos químicos."
> → **Classe 1** (interferência em exame, sem risco clínico grave)

### Classe 2 — GRAVE_CONTRAINDICADA

**Interação com risco significativo: contraindicação explícita, risco de morte, evento adverso grave, ou dano orgânico permanente.**

Palavras-chave e frases típicas:
- "contraindicado", "contraindicação"
- "não deve ser administrado", "não administrar"
- "nunca associar"
- "risco de morte", "fatal"
- "arritmia fatal", "parada cardíaca"
- "insuficiência renal aguda"
- "rabdomiólise"
- "hemorragia grave"
- "síndrome de Stevens-Johnson"
- "toxicidade grave", "hepatotoxicidade"
- "interação severa"
- "risco de vida"
- "contraindicação absoluta"

**Exemplo real (Fonte 2, `zocor.txt`):**
> "É muito importante informar ao seu médico se você for tomar Zocor associado a [...] pois o risco de problemas musculares nessa situação é maior. [...] Em raras ocasiões, problemas musculares podem ser graves, incluindo rompimento muscular, resultando em dano renal que pode ser fatal."
> → **Classe 2** (sinvastatina + itraconazol/cetoconazol — risco de rabdomiólise fatal)

**Exemplo real (Fonte 1, `105830895_amoxicilina_profissional.txt`):**
> "Há relatos de reações de hipersensibilidade graves e ocasionalmente fatais (incluindo reações adversas severas anafilactoides e cutâneas) em pacientes que receberam tratamento com penicilinas."
> → **Classe 2** (amoxicilina em pacientes com histórico de hipersensibilidade — contraindicação absoluta)

---

## 4. Árvore de Decisão para Classificação

```
1. O texto menciona o medicamento_outro?
   ├── NÃO → NÃO ANOTAR (não há par relevante)
   └── SIM → Continue

2. O texto descreve alguma interação ou efeito da combinação?
   ├── NÃO → O texto apenas lista os medicamentos sem descrever efeito?
   │   ├── SIM → NÃO ANOTAR (informação insuficiente)
   │   └── NÃO → Continue
   └── SIM → Continue

3. Qual a gravidade descrita?
   ├── Afirma explicitamente que NÃO há interação / é seguro?
   │   └── SIM → CLASSE 0 (SEM_INTERACAO)
   │
   ├── Contém "contraindicado", "risco de morte", "fatal", "não administrar",
   │   "rabdomiólise", "hemorragia grave", "Stevens-Johnson"?
   │   └── SIM → CLASSE 2 (GRAVE_CONTRAINDICADA)
   │
   ├── Contém "monitorar", "ajustar", "cautela", "pode aumentar/reduzir",
   │   "precaução", "recomenda-se acompanhamento"?
   │   └── SIM → CLASSE 1 (LEVE_MODERADA)
   │
   └── Nenhum dos anteriores?
       └── NÃO ANOTAR (caso ambíguo, deixar para curadoria manual)
```

---

## 5. Procedimento de Anotação Passo a Passo

### Passo 1: Localizar os arquivos

Os dados já processados estão em:
```
C:\workspace\python\python-processador-bulas\data\pruned\
```

O JSONL com metadados está em:
```
C:\workspace\python\python-processador-bulas\data\pruned\bulas_pruned.jsonl
```

### Passo 2: Anotar Fonte 2 primeiro (mais fácil — já estruturado)

A Fonte 2 é mais simples porque o bloco `INTERAÇÃO MEDICAMENTOSA?` já isola o texto relevante.

**Para cada arquivo da Fonte 2:**

1. Abra `data/pruned/fonte2/<medicamento>.txt`
2. Localize o bloco `[P: INTERAÇÃO MEDICAMENTOSA?]`
3. Leia a resposta completa (campo `R:`)
4. Extraia o **medicamento alvo** do nome do arquivo
   - Ex: `zarator.txt` → medicamento alvo = `atorvastatina` (nome do princípio ativo)
   - Ex: `zocor.txt` → medicamento alvo = `sinvastatina`
   - Ex: `zyloric.txt` → medicamento alvo = `alopurinol`
5. Identifique cada **medicamento mencionado** na resposta
6. Para cada medicamento_outro, crie um par:
   - `medicamento_alvo`: nome do arquivo (princípio ativo)
   - `medicamento_outro`: nome do medicamento mencionado
   - `contexto`: o parágrafo ou sentença que menciona a interação
   - `classe`: 0, 1 ou 2 conforme as regras acima
   - `fonte`: `"fonte2"`
   - `origem`: `"automatica"` (se classificado por heurística) ou `"manual"`

**📋 Exemplo de anotação da Fonte 2 (`zarator.txt`):**

| medicamento_alvo | medicamento_outro | contexto | classe |
|---|---|---|---|
| atorvastatina | ciclosporina | "Miopatia devido à lesão dos músculos pode ocorrer em pacientes que usam Zarator, sendo mais frequentes naqueles que usam também ciclosporina, fibratos, niacina ou antifúngicos azólicos." | 1 |
| atorvastatina | eritromicina | "A administração concomitante de Zarator com medicamentos inibidores do citocromo P450 3A4 (por ex., ciclosporina, eritromicina/claritromicina, inibidores da protease, cloridrato de diltiazem, suco de grapefruit) pode alterar a quantidade de atorvastatina no sangue." | 1 |
| atorvastatina | varfarina | "São conhecidas outras interações medicamentosas, avise seu médico se você fizer uso de antiácidos, colestipol, contraceptivos orais, varfarina, ácido fusídico." | 1 |

> ⚠️ Nota: No exemplo acima, todos foram classe 1 porque o texto do Zarator não usa termos como "contraindicado" ou "fatal" para esses medicamentos específicos. O risco de rabdomiólise é mencionado em outra parte mas não atribuído diretamente a um medicamento específico.

### Passo 3: Anotar Fonte 1 (mais trabalhoso — texto corrido)

A Fonte 1 requer mais atenção porque as interações estão diluídas em seções maiores.

**Para cada arquivo da Fonte 1:**

1. Abra `data/pruned/fonte1/<id>_<medicamento>_<versao>.txt`
2. Localize a seção `## INTERAÇÕES MEDICAMENTOSAS` (pode aparecer como `## INTERAÇÕES  MEDICAMENTOSAS` com espaços extras)
3. Se não encontrar, procure fallbacks: `## ADVERTÊNCIAS E PRECAUÇÕES`, `## CONTRAINDICAÇÕES`
4. Extraia o **medicamento alvo** do nome do arquivo:
   - Ex: `105830895_amoxicilina_profissional.txt` → `amoxicilina`
   - Ex: `100380098_captopril_profissional.txt` → `captopril`
5. Leia parágrafo por parágrafo. Cada parágrafo que menciona outro medicamento + descreve um efeito = um par potencial.
6. Classifique conforme as regras da seção 3.

**📋 Exemplo de anotação da Fonte 1 (`105830895_amoxicilina_profissional.txt`):**

| medicamento_alvo | medicamento_outro | contexto | classe |
|---|---|---|---|
| amoxicilina | probenecida | "A probenecida reduz a secreção tubular renal da amoxicilina. No uso concomitante com amoxicilina, pode haver aumento dos níveis de amoxicilina no sangue e no prolongamento dessa alteração." | 1 |
| amoxicilina | contraceptivos orais | "Da mesma forma que outros antibióticos, amoxicilina pode afetar a flora intestinal, levando a menor reabsorção de estrógenos, e reduzir a eficácia dos contraceptivos orais." | 1 |
| amoxicilina | alopurinol | "A administração concomitante de alopurinol durante o tratamento com amoxicilina pode aumentar a probabilidade de reações alérgicas da pele." | 1 |
| amoxicilina | varfarina | "Na literatura existem casos raros de INR aumentada em pacientes mantidos com acenocumarol ou varfarina, ao receberem um curso de tratamento com amoxicilina. Se a coadministração é necessária, o tempo de protrombina ou INR deve ser cuidadosamente monitorado." | 1 |
| amoxicilina | acenocumarol | "(mesmo contexto acima)" | 1 |

### Passo 4: Priorizar arquivos com maior densidade de interações

Use o JSONL para filtrar bulas mais promissoras:

- **Fonte 1**: Priorize arquivos com `secoes_mantidas` contendo `"INTERAÇÕES MEDICAMENTOSAS"` e `status = "ok"`
- **Fonte 2**: Priorize arquivos com `blocos_mantidos` contendo `"INTERAÇÃO MEDICAMENTOSA?"` — praticamente todos têm
- **Evite**: Arquivos com `status = "fallback"` e `reducao_percentual > 95%` (provavelmente não têm seção de interações — a poda removeu quase tudo)

**Comando para filtrar os melhores candidatos (PowerShell):**
```powershell
# Fonte 1: apenas bulas que contêm INTERAÇÕES MEDICAMENTOSAS (no JSONL)
Select-String -Path "C:\workspace\python\python-processador-bulas\data\pruned\bulas_pruned.jsonl" -Pattern '"INTERAÇÕES MEDICAMENTOSAS"' | Select-Object -First 200
```

### Passo 5: Registrar as anotações

Crie um CSV com as seguintes colunas:

```csv
medicamento_alvo,medicamento_outro,contexto,classe,fonte,origem
amoxicilina,probenecida,"A probenecida reduz a secreção tubular renal da amoxicilina...",1,fonte1,manual
amoxicilina,alopurinol,"A administração concomitante de alopurinol durante o tratamento...",1,fonte1,manual
atorvastatina,ciclosporina,"Miopatia devido à lesão dos músculos pode ocorrer...",1,fonte2,automatica
```

Salve em:
```
C:\workspace\python\projeto-2-modulo-1-pos\data\anotacoes\manuais.csv
```

---

## 6. Estratégia de Amostragem

Para maximizar a qualidade do fine-tuning com esforço mínimo de anotação:

### Meta: ~1.500 pares anotados

| Fonte | Quantidade | Origem | Prioridade |
|---|---|---|---|
| **Automáticos (Fonte 2)** | ~1.000 | Heurística (script) | ⚡ Rápido — fazer primeiro |
| **Manuais (Fonte 1)** | ~500 | Curadoria humana | 🎯 Foco em casos ambíguos |

### Como selecionar os 500 manuais da Fonte 1

1. **Filtrar bulas com `INTERAÇÕES MEDICAMENTOSAS`** no campo `secoes_mantidas` do JSONL
2. **Priorizar versão `_profissional`** sobre `_paciente` (texto mais detalhado)
3. **Priorizar bulas com múltiplas interações**: parágrafos com 2+ medicamentos mencionados
4. **Garantir diversidade**: incluir antibióticos, anti-hipertensivos, AINEs, anticoagulantes, antidiabéticos, psicotrópicos
5. **Garantir balanceamento**: ~200 classe 0, ~150 classe 1, ~150 classe 2

---

## 7. Medicamentos Prioritários para Anotação

Baseado na frequência nas bulas e relevância clínica:

### Alta prioridade (interações frequentes e graves)

| Medicamento | Classe terapêutica | Por que priorizar |
|---|---|---|
| Varfarina | Anticoagulante | Muitas interações graves (INR alterado, sangramento) |
| Amoxicilina | Antibiótico | Alta prevalência, interações com anticoagulantes |
| Sinvastatina / Atorvastatina | Estatina | Risco de rabdomiólise com antifúngicos/macrolídeos |
| Captopril / Enalapril | IECA | Interações com diuréticos, AINEs, potássio |
| Metformina | Antidiabético | Interações com álcool, contraste iodado |
| Fluoxetina / Sertralina | ISRS | Síndrome serotoninérgica com outros serotoninérgicos |
| Omeprazol | IBP | Reduz absorção de antifúngicos, clopidogrel |
| Ibuprofeno / Dipirona | AINE | Interações com anticoagulantes, anti-hipertensivos |

### Média prioridade

| Medicamento | Classe terapêutica |
|---|---|
| Losartana | BRA |
| Levotiroxina | Hormônio tireoidiano |
| Clonazepam / Diazepam | Benzodiazepínico |
| Carbamazepina | Anticonvulsivante |
| Alopurinol | Anti-gota |
| Cetoconazol / Fluconazol | Antifúngico |

---

## 8. Formato do CSV de Saída

```csv
medicamento_alvo,medicamento_outro,contexto,classe,fonte,origem,confianca
amoxicilina,probenecida,"A probenecida reduz a secreção tubular renal...",1,fonte1,manual,1.0
amoxicilina,varfarina,"Na literatura existem casos raros de INR aumentada...",1,fonte1,manual,0.9
atorvastatina,ciclosporina,"Miopatia devido à lesão dos músculos...",1,fonte2,automatica,0.85
sinvastatina,itraconazol,"...risco de problemas musculares...pode ser fatal",2,fonte2,automatica,0.95
```

**Colunas:**
| Coluna | Tipo | Descrição |
|---|---|---|
| `medicamento_alvo` | string | Princípio ativo da bula (normalizado: lowercase, sem acentos) |
| `medicamento_outro` | string | Medicamento com o qual interage (normalizado) |
| `contexto` | string | Sentença ou parágrafo que descreve a interação (até 500 caracteres) |
| `classe` | int | 0 (SEM), 1 (LEVE_MODERADA), 2 (GRAVE_CONTRAINDICADA) |
| `fonte` | string | `"fonte1"` ou `"fonte2"` |
| `origem` | string | `"automatica"` (heurística) ou `"manual"` (curadoria humana) |
| `confianca` | float | 0.0 a 1.0 — 1.0 para anotação manual; ~0.7-0.9 para heurística automática |

---

## 9. Dicas Práticas

### Normalização de nomes de medicamentos

- **Sempre lowercase**: `Varfarina` → `varfarina`
- **Remover acentos**: `ácido acetilsalicílico` → `acido acetilsalicilico`
- **Remover sufixos de sal**: `cloridrato de ciprofloxacino` → `ciprofloxacino`
- **Nomes comerciais → princípio ativo** (quando souber): `Zarator` → `atorvastatina`, `Zocor` → `sinvastatina`
- **Manter consistência**: se usou `amoxicilina` em um par, use `amoxicilina` em todos

### Contextos que NÃO devem ser anotados

- ❌ Parágrafos que mencionam um medicamento mas não descrevem interação com o alvo
- ❌ Listas de medicamentos sem descrição do efeito da interação
- ❌ Menções a "medicamentos da classe X" sem nomear medicamentos específicos
- ❌ Interações com alimentos (ex: "suco de grapefruit") — a menos que explicitamente solicitado
- ❌ Interações com álcool — mesma regra

### Quando marcar confiança < 1.0 em anotações manuais

- O texto é ambíguo (ex: "pode haver interação" sem detalhes)
- O parágrafo menciona o medicamento de passagem
- A classificação depende de conhecimento clínico externo (não está explícita na bula)

Use `confianca = 0.7` para casos duvidosos. Isso permite que o script de fine-tuning dê menos peso a esses exemplos.

---

## 10. Checklist de Qualidade

Antes de finalizar o dataset:

- [ ] Total de pares: ≥ 1.200 (ideal: 1.500)
- [ ] Distribuição de classes: classe 0 ≥ 25%, classe 1 ≥ 25%, classe 2 ≥ 15%
- [ ] Nomes de medicamentos normalizados (lowercase, sem acentos)
- [ ] Contextos com 30 a 500 caracteres
- [ ] Nenhum par duplicado (`medicamento_alvo + medicamento_outro + contexto` idêntico)
- [ ] Nenhum `medicamento_alvo == medicamento_outro`
- [ ] Pelo menos 8 classes terapêuticas diferentes representadas
- [ ] CSV carrega sem erro: `pd.read_csv("manuais.csv")`
- [ ] Coluna `classe` contém apenas 0, 1 ou 2
- [ ] Coluna `fonte` contém apenas `"fonte1"` ou `"fonte2"`

---

## 11. Referência Rápida de Classificação

| Se a bula diz... | Classe |
|---|---|
| "não há interação", "não foram observadas", "seguro", "pode ser usado" | **0** |
| "monitorar", "ajustar", "cautela", "precaução", "pode aumentar/reduzir" | **1** |
| "contraindicado", "não administrar", "risco de morte", "fatal", "interação severa" | **2** |
| Apenas lista o medicamento sem descrever efeito | **NÃO ANOTAR** |
| Não menciona interação com o medicamento_alvo | **NÃO ANOTAR** |
