# Fase 7: Notebook 04 — Inferência Local vs Remota

**Objetivo:** Comparar execução local (GPT4All Phi-3-mini) vs remota (OpenAI GPT-4o-mini) em 5 dimensões: qualidade, latência, custo, privacidade e controle. Justificar a escolha para o domínio da saúde.

**Dependências:** Fase 5 (classe LLMProvider, 200 pares anotados).

**Rubricas cobertas:** Rubrica 4 — todos os 5 itens.

---

## Estrutura do Notebook

```
c04_inferencia_local_ou_remota.ipynb
├── 7.1  Setup: instanciar ambos os backends
├── 7.2  Teste de qualidade (acurácia)
├── 7.3  Teste de latência
├── 7.4  Análise de custo
├── 7.5  Análise de privacidade
├── 7.6  Análise de controle e disponibilidade
└── 7.7  Conclusão e recomendação
```

---

## Tarefas

### 7.1 Setup: ambos os backends

- [ ] Instanciar `LLMProvider("openai")` — requer `OPENAI_API_KEY` no `.env`
- [ ] Instanciar `LLMProvider("gpt4all")` — download automático do modelo GGUF na primeira execução
- [ ] Verificar que ambos respondem a um prompt simples: `"Classifique: Amoxicilina + Ibuprofeno. Responda JSON."`
- [ ] Célula Markdown: apresentação dos dois backends, arquitetura de cada um

### 7.2 Teste de qualidade (acurácia + F1)

- [ ] Usar os mesmos 200 pares da Fase 5
- [ ] Prompt zero-shot (template base), idêntico para ambos
- [ ] Métricas lado a lado:

| Métrica | GPT4All (Phi-3-mini) | OpenAI (GPT-4o-mini) |
|---|---|---|
| Acurácia | X% | X% |
| F1 Classe 0 | X% | X% |
| F1 Classe 1 | X% | X% |
| F1 Classe 2 (GRAVE) | X% | X% |
| F1 Macro | X% | X% |
| % JSON Válido | X% | X% |

- [ ] Matriz de confusão lado a lado (duas matrizes)
- [ ] Análise: onde o modelo local erra mais? (classes graves, textos longos, nuance)

### 7.3 Teste de latência

- [ ] 20 consultas idênticas em ambos
- [ ] Métricas: média, mediana, p95, p99

| Backend | Média (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| GPT4All (CPU) | X | X | X |
| OpenAI API | X | X | X |

- [ ] Gráfico de barras: latência por query (20 pontos, duas séries)
- [ ] Célula Markdown: GPT4All em CPU é ~2-5x mais lento, mas sem latência de rede. OpenAI depende de internet.

### 7.4 Análise de custo

- [ ] **GPT4All**: custo fixo (hardware já existe) + eletricidade (~R$ 0,05/hora). Custo marginal ≈ R$ 0,00
- [ ] **OpenAI GPT-4o-mini**: $0.15/1M input + $0.60/1M output tokens
  - Estimar: ~500 tokens input + ~150 tokens output por consulta
  - Custo por consulta: ~$0.000075 + ~$0.00009 = ~$0.000165
- [ ] Tabela:

| Cenário | GPT4All (Local) | OpenAI (Remoto) |
|---|---|---|
| 1.000 consultas | R$ 0,00 | R$ 0,92 |
| 10.000 consultas | R$ 0,00 | R$ 9,20 |
| 100.000 consultas | R$ 0,00 | R$ 92,00 |
| Custo fixo (GPU) | R$ 1.500+ | R$ 0,00 |

- [ ] Célula Markdown: para uso esporádico (< 1.000/mês), API é mais barata que comprar GPU. Para produção (10.000+/mês), local se paga em ~18 meses.

### 7.5 Análise de privacidade

- [ ] **GPT4All local**:
  - ✅ Dados nunca saem da máquina
  - ✅ Compatível com LGPD/HIPAA
  - ✅ Sem dependência de política de privacidade de terceiros
- [ ] **OpenAI API**:
  - ⚠️ Dados enviados para servidores nos EUA
  - ⚠️ Retidos por 30 dias (política atual da OpenAI)
  - ✅ API não usa dados para treino (opt-out padrão para API)
  - ⚠️ LGPD: transferência internacional exige contrato e garantias
- [ ] **Bulas da ANVISA são públicas** → risco baixo. Mas consultas revelam:
  - Condições de saúde do usuário
  - Medicamentos que está tomando
  - Potenciais comorbidades
- [ ] Célula Markdown: para deploy real em hospital/clínica, modelo local é obrigatório. Para protótipo acadêmico, API remota é aceitável.

### 7.6 Análise de controle e disponibilidade

- [ ] **GPT4All**:
  - ✅ Controle total sobre versão do modelo
  - ✅ Funciona offline (consultas em áreas sem internet)
  - ✅ Sem risco de depreciação de API ou mudança de preço
  - ⚠️ Atualização manual do modelo
- [ ] **OpenAI**:
  - ✅ Sempre o modelo mais recente
  - ✅ Zero manutenção
  - ⚠️ Depende de internet
  - ⚠️ Risco de outage (já ocorreu)
  - ⚠️ Modelo pode ser depreciado (ex: GPT-3.5-turbo)

### 7.7 Conclusão e recomendação

- [ ] Tabela resumo com 5 dimensões:

| Dimensão | GPT4All Local | OpenAI Remoto | Vencedor |
|---|---|---|---|
| Qualidade (F1) | ⭐⭐ | ⭐⭐⭐ | OpenAI |
| Latência | ⭐⭐ | ⭐⭐⭐ | OpenAI |
| Custo (>10K/mês) | ⭐⭐⭐ | ⭐ | Local |
| Privacidade | ⭐⭐⭐ | ⭐ | Local |
| Controle | ⭐⭐⭐ | ⭐⭐ | Local |

- [ ] **Recomendação para o projeto:** Notebooks usam OpenAI (qualidade), mas o pipeline RAG (Fase 8) suporta ambos via `LLMProvider`. O código comprova que a troca é transparente.
- [ ] **Recomendação para produção:** GPT4All local para garantir privacidade do paciente. Custo se paga em escala.

---

## Artefatos Produzidos

```
c04_inferencia_local_ou_remota.ipynb
```

---

## Verificação

- [ ] Notebook executa do início ao fim sem erros (exceto células OpenAI se sem API key — tratar com graceful skip)
- [ ] Ambos os backends respondem com JSON válido em > 80% dos casos
- [ ] Tabela de qualidade preenchida com dados reais (não placeholders)
- [ ] Análise de privacidade cita LGPD e implicações para saúde
- [ ] Célula Markdown com recomendação clara e justificada
- [ ] Commit: `git add c04_inferencia_local_ou_remota.ipynb && git commit -m "feat: Fase 7 — Notebook 04: inferência local vs remota comparada"`
