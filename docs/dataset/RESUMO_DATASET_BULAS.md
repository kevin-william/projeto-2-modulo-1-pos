# Resumo do Dataset — Bulas de Medicamentos

**Projeto:** Detector de Interações Medicamentosas com LLMs e RAG  
**Total de arquivos:** 5.960  
**Localização:** `data/bulas/`

---

## 1. Fontes e Quantidades

| Fonte | Descrição | Quantidade | Formato | Nomenclatura |
|---|---|---|---|---|
| **Fonte 1** | Bulas oficiais da ANVISA (profissional + paciente) | 4.978 | Texto corrido com seções demarcadas por cabeçalhos | `12345_medicamento_versao.txt` |
| **Fonte 2** | Bulas do site Consultaremedios (Q&A estruturado) | 982 | 16 blocos de Perguntas & Respostas | `medicamento-profissional.txt` |

---

## 2. Estrutura dos Arquivos

### Fonte 1 (ANVISA) — Texto Corrido

```
# MEDICAMENTO (princípio_ativo)

## APRESENTAÇÕES
Comprimidos de 500 mg...

## COMPOSIÇÃO
Cada comprimido contém...

## INDICAÇÕES
Este medicamento é indicado para...

## CONTRAINDICAÇÕES
Hipersensibilidade ao princípio ativo...

## INTERAÇÕES MEDICAMENTOSAS
A administração concomitante com Metotrexato pode aumentar a toxicidade...

## REAÇÕES ADVERSAS
Podem ocorrer náuseas, diarreia...
```

**Características:**
- Seções em MAIÚSCULAS ou Capitalizadas como cabeçalhos
- Versão `_paciente`: texto simplificado, seções como "O que devo saber antes de usar?"
- Versão `_profissional`: texto técnico completo, seções detalhadas
- Tamanho: 2.000 a 15.000 tokens por bula (texto completo)

### Fonte 2 (Consultaremedios) — Q&A Estruturado

```
1. PARA QUE ESTE MEDICAMENTO É INDICADO?
Resposta...

2. COMO ESTE MEDICAMENTO FUNCIONA?
Resposta...

...

10. INTERAÇÃO MEDICAMENTOSA?
Resposta sobre interações...

11. QUAIS OS MALES QUE ESTE MEDICAMENTO PODE CAUSAR?
Resposta...
```

**Características:**
- 16 perguntas fixas numeradas
- Bloco 10: `INTERAÇÃO MEDICAMENTOSA?` — foco principal do projeto
- Bloco 3: `COMPOSIÇÃO?` — informação complementar (excipientes)
- Respostas em parágrafos, podendo estar truncadas
- Tamanho: 500 a 3.000 tokens por bula

---

## 3. Seções de Interesse por Fonte

### Fonte 1

| Versão | Seções Prioritárias | Seções de Fallback |
|---|---|---|
| `_profissional` | "INTERAÇÕES MEDICAMENTOSAS" | "PRECAUÇÕES", "CONTRAINDICAÇÕES", "ADVERTÊNCIAS" |
| `_paciente` | "INTERAÇÕES MEDICAMENTOSAS" | "O QUE DEVO SABER ANTES DE USAR?", "REAÇÕES ADVERSAS" |

### Fonte 2

| Prioridade | Bloco |
|---|---|
| Primário | 10. `INTERAÇÃO MEDICAMENTOSA?` |
| Secundário | 3. `COMPOSIÇÃO?` (para excipientes) |

---

## 4. Tipos de Medicamentos nas Bulas

### Fonte 1 (ANVISA)
- **Predominante:** Princípios ativos (Amoxicilina, Dipirona, Losartana, Metformina, etc.)
- **Formato:** Nome do arquivo começa com número + underscore + nome do princípio ativo

### Fonte 2 (Consultaremedios)
- **Predominante:** Nomes comerciais (AAS Protect, Novalgina, Cozaar, Glifage, etc.)
- **Formato:** Nome do arquivo começa com letra + underscore + nome comercial

**Implicação para NER:** O modelo `clinicalnerpt-chemical` foi treinado com ambos os tipos, permitindo extrair tanto princípios ativos quanto nomes comerciais de forma unificada.

---

## 5. Exemplo de Trecho com Interação (Fonte 1)

```
INTERAÇÕES MEDICAMENTOSAS

Metotrexato: a administração concomitante de penicilinas, incluindo 
amoxicilina, com metotrexato (usado em altas doses para tratamento de 
câncer e artrite reumatoide) pode aumentar a toxicidade do metotrexato, 
devido à redução na sua secreção tubular renal. Portanto, o uso 
concomitante deve ser evitado.

Anticoagulantes orais: o uso concomitante de amoxicilina e 
anticoagulantes orais (ex.: varfarina) pode prolongar o tempo de 
protrombina. Recomenda-se monitoramento frequente.
```

**Entidades extraíveis:** Amoxicilina, Metotrexato, Varfarina  
**Classificação esperada:** Metotrexato → GRAVE_CONTRAINDICADA; Varfarina → LEVE_MODERADA

---

## 6. Exemplo de Trecho com Interação (Fonte 2)

```
10. INTERAÇÃO MEDICAMENTOSA?

O uso concomitante de AAS Protect com anticoagulantes (como varfarina) 
pode aumentar o risco de sangramentos. Deve-se evitar o uso com outros 
AINEs (como ibuprofeno) devido ao risco aumentado de úlcera gástrica. 

Não há interações clinicamente relevantes com paracetamol quando 
utilizado nas doses recomendadas.
```

**Entidades extraíveis:** AAS Protect (ácido acetilsalicílico), Varfarina, Ibuprofeno, Paracetamol  
**Classificação esperada:** Varfarina → LEVE_MODERADA; Ibuprofeno → LEVE_MODERADA; Paracetamol → SEM_INTERACAO

---

## 7. Estatísticas do Dataset

| Métrica | Fonte 1 | Fonte 2 | Total |
|---|---|---|---|
| Arquivos | 4.978 | 982 | 5.960 |
| Profissional | ~2.500 | 982 | ~3.482 |
| Paciente | ~2.478 | 0 | ~2.478 |
| Tamanho médio (tokens) | ~5.000 | ~1.500 | — |
| Seções de interação (estimado) | ~85% contêm | ~95% contêm | — |

---

## 8. Localização dos Dados

```
data/bulas/
├── fonte1/    # 4.978 arquivos .txt (ANVISA)
└── fonte2/    # 982 arquivos .txt (Consultaremedios)
```

**Nota:** Os dados reais NÃO são versionados no git (estão no `.gitignore`). Apenas metadados e amostras anonimizadas são incluídos no repositório.
