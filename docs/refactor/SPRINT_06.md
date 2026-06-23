# Sprint 6 — Finalização (README + Relatório PDF)

**Objetivo:** Produzir os artefatos finais de entrega: README.md com instruções
de reprodução e relatório técnico em PDF cobrindo todas as 26 seções obrigatórias.

**Duração:** 3-4 horas  
**Commits:** 3 atômicos  
**Rubricas cobertas:** Rubrica 5 (2 itens restantes: solução documentada, análise crítica)

---

## 1. README.md

### 1.1 Estrutura completa

```markdown
# Detector de Interações Medicamentosas com LLMs e RAG

Sistema cognitivo que recebe consultas em linguagem natural sobre interações
medicamentosas e retorna classificação fundamentada em 5.960 bulas reais
(ANVISA e Consultaremedios).

**100% local.** Nenhuma API externa necessária. Todos os modelos rodam na sua máquina.

## Requisitos

-   Python 3.9 ou superior
-   8 GB de RAM (16 GB recomendado — o GPT4All consome ~4.5 GB)
-   GPU NVIDIA com 6 GB de VRAM (opcional — funciona em CPU, apenas mais lento)
-   6 GB de espaço em disco (modelos .gguf + embeddings)
-   Windows, Linux ou macOS

## Instalação

```bash
# 1. Clonar o repositório
git clone <repositorio>
cd projeto-2-modulo-1-pos

# 2. Criar e ativar ambiente virtual
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# ou: venv\Scripts\activate    # Windows (CMD)

# 3. Instalar PyTorch com suporte CUDA (se tiver GPU NVIDIA)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Instalar demais dependências
pip install -r requirements.txt
```

## Dados

Copie as bulas pré-processadas do projeto de referência:

```bash
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte1 data/bulas/
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte2 data/bulas/
mkdir logs
```

## Execução

Abra os cadernos Jupyter na ordem abaixo. Cada caderno é autossuficiente
e gera seu próprio arquivo de log em `logs/`.

| Ordem | Caderno | Tempo estimado | Conteúdo |
|---|---|---|---|
| 1 | `c01_modelos_llm.ipynb` | 5 min | HuggingFace, pipelines, AutoModel, NER |
| 2 | `c02_engenharia_prompt.ipynb` | 10 min | GPT4All, 3 técnicas de prompt, parsing JSON |
| 3 | `c03_embeddings_busca.ipynb` | 8 min | Embeddings, FAISS, busca híbrida |
| 4 | `c04_inferencia_local.ipynb` | 8 min | Comparação GPT4All direct vs API vs heurística |
| 5 | `c05_pipeline_rag.ipynb` | 15 min | Pipeline RAG completo com 8 consultas demo |

```bash
jupyter notebook
```

Dentro do Jupyter, para cada caderno: **Kernel → Restart & Run All**.

## Logs

Todos os cadernos geram logs com timestamps em `logs/`. Para acompanhar
a execução em tempo real:

```bash
tail -f logs/caderno_05_pipeline.log
```

## Estrutura do Projeto

```
├── c01_modelos_llm.ipynb              # HF pipelines + logging
├── c02_engenharia_prompt.ipynb        # GPT4All + 3 técnicas de prompt
├── c03_embeddings_busca.ipynb         # FAISS + busca híbrida
├── c04_inferencia_local.ipynb         # GPT4All direct vs API vs heurística
├── c05_pipeline_rag.ipynb             # Pipeline RAG completo
├── README.md
├── requirements.txt
├── .gitignore
├── data/bulas/fonte1/                 # 4.978 bulas ANVISA
├── data/bulas/fonte2/                 # 982 bulas Consultaremedios
├── logs/                              # Logs de execução
└── docs/
    ├── PLANO_REBOOT.md
    └── refactor/
        ├── SPRINT_01.md
        ├── SPRINT_02.md
        ├── SPRINT_03.md
        ├── SPRINT_04.md
        ├── SPRINT_05.md
        └── SPRINT_06.md
```

## Solução de Problemas

| Problema | Solução |
|---|---|
| `CUDA not available` | Reinstale PyTorch com CUDA (veja instrução acima) ou execute em CPU |
| `Out of memory` | Reduza `maximo_arquivos` no caderno 05 para 100 |
| `GPT4All model not found` | O download automático será iniciado na primeira execução |
| `API server connection refused` | O caderno 02 fará fallback automático para heurística |
| `ModuleNotFoundError` | Execute `pip install -r requirements.txt` novamente |

## Créditos

-   **Aluno:** Kevin Rodrigues
-   **Disciplina:** Sistemas Cognitivos com Large Language Models
-   **Corpus:** 5.960 bulas da ANVISA e Consultaremedios
-   **Modelos:** clinicalnerpt-chemical, bert-base-portuguese-cased, GPT4All Llama-3-8B
```

---

## 2. Relatório PDF — Mapeamento das 26 Seções

### 2.1 Template de cada seção com referência ao código

| # | Seção | Conteúdo | Evidência |
|---|---|---|---|
| 1 | Nome do aluno | Kevin Rodrigues | — |
| 2 | Nome da disciplina | Sistemas Cognitivos com Large Language Models | — |
| 3 | Título do projeto | Detector de Interações Medicamentosas com LLMs, NER e RAG | — |
| 4 | Descrição do problema | Profissionais de saúde precisam verificar interações medicamentosas rapidamente. Bulas são extensas (até 10 mil tokens) e técnicas. A solução extrai automaticamente interações e as classifica. | Contextualização no caderno 05, célula 1 |
| 5 | Descrição do corpus | 5.960 bulas brasileiras: 4.978 da ANVISA (texto corrido, seções `##`) e 982 do Consultaremedios (16 blocos Q&A). Seções de interesse: "Interações Medicamentosas", "Precauções", "Contraindicações". | `docs/dataset/RESUMO_DATASET_BULAS.md` (projeto de referência) |
| 6 | Justificativa para uso de LLMs | NER (variabilidade de nomes de medicamentos — inviável com regex), classificação (compreensão semântica além de palavras-chave), RAG (fundamentação em documentos reais para reduzir alucinação) | Caderno 01, células 7-8 (NER) e caderno 05, célula 1 |
| 7 | Modelos, APIs ou ferramentas | Tabela completa: clinicalnerpt-chemical (NER), bert-base-portuguese-cased (embeddings), GPT4All Llama-3-8B (LLM local), FAISS (vector store) | `docs/PLANO_REBOOT.md` seção 3.3 |
| 8 | Tarefas NLP implementadas | NER, classificação de texto, geração de embeddings, geração de texto (LLM) | Caderno 01 (células 4, 6, 8), caderno 05 (célula 5) |
| 9 | Estratégia de prompting | 3 técnicas comparadas: zero-shot, few-shot (3 exemplos), cadeia de pensamento (3 etapas de raciocínio) | Caderno 02, células 4-6 |
| 10 | Prompts utilizados e versões testadas | Reprodução integral dos 3 prompts. Iterações: v1 (sem evidencia) → v2 (com evidencia) → v3 (temperature 0.7→0.1) | Caderno 02, células 4-6 + análise na célula 8 |
| 11 | Estratégia de avaliação dos prompts | 30 pares com gabarito (10 por classe). Métricas: acurácia, F1 por classe, % JSON válido, latência | Caderno 02, célula 8 |
| 12 | Uso de JSON, parsing ou saída estruturada | `analisar_resposta_json()` com 3 estratégias: json.loads(), remoção de markdown, regex fallback | Caderno 02, célula 7 |
| 13 | Modelos de embeddings utilizados | BERT português (768d) vs MiniLM (384d) comparados | Caderno 03, célula 7 |
| 14 | Estratégia de busca vetorial | FAISS IndexFlatIP + cosseno + BM25 híbrida (peso_cosseno=0.3) | Caderno 03, células 4-6 |
| 15 | Exemplos de consultas e documentos recuperados | 10 consultas de teste com top-5 resultados cada | Caderno 03, célula 5 |
| 16 | Estratégia de execução | 100% local. GPT4All com fallback em 3 camadas: ligação direta → servidor API → heurística. Comparação detalhada. | Caderno 04, células 2-8 |
| 17 | Justificativa sobre privacidade, custo, latência e controle | Tabela 5 dimensões comparando os 3 backends. Análise LGPD. Custo zero (sem APIs pagas). | Caderno 04, células 5-8 |
| 18 | Descrição do pipeline RAG | Diagrama ASCII + fluxo NER → FAISS → GPT4All → JSON. Código completo da classe `PipelineInteracao`. | Caderno 05, células 1 e 5 |
| 19 | Estratégia de chunking | 3 estratégias comparadas: sentenças (recall@5=0.78), parágrafos de 3 (0.65), parágrafos de 5 (0.55). Sentenças maximizam precisão. | Caderno 05, célula 9 |
| 20 | Vector store | FAISS IndexFlatIP — justificativa: sem dependências externas, em memória, inner product = cosseno com vetores normalizados | Caderno 03, célula 4 e 8 |
| 21 | Exemplos de consultas e respostas | 8 consultas demo com JSON completo de saída: grave, leve, sem, não encontrado, ambígua, múltiplos, nome comercial, não-medicamento | Caderno 05, célula 7 |
| 22 | Análise de respostas com e sem contexto | 5 pares comparados: modo A (zero-shot sem bulas) vs modo B (RAG com FAISS). Exemplo de alucinação reduzida | Caderno 05, célula 8 |
| 23 | Análise de falhas do pipeline | 3 cenários: NER não reconhece medicamento → fallback para busca textual; trechos irrelevantes → refinar query; classificação incorreta → mostrar confiança | Caderno 05, célula 10 |
| 24 | Riscos de segurança | Prompt injection (demonstrado + sanitização), vazamento de contexto (mínimo — bulas públicas), envenenamento de dados, alucinação | Caderno 05, células 11-12 |
| 25 | Controles propostos | Sanitização de entrada, validação de schema JSON, threshold de confiança, logging de auditoria, isolamento local (sem rede) | Caderno 05, células 4 e 12 |
| 26 | Instruções de reprodução | Remissão ao README.md | README.md |
| 27 | Limitações da solução | 6 limitações: NER treinado em corpus geral, cobertura de 5.960 bulas, sem 3+ medicamentos, sem interações alimento/exame, qualidade do GPT4All vs APIs cloud, busca puramente semântica | Caderno 05, célula 13 |
| 28 | Melhorias futuras | Fine-tuning do BioBERTpt, interface web, Knowledge Graph para sinônimos, atualização automática, suporte a interações complexas | Caderno 05, célula 13 |

### 2.2 Nome do arquivo

```
kevin_rodrigues_sistemas-cognitivos-linguagem-natural_aplicacoes-llms.pdf
```

### 2.3 Checklist de verificação antes da entrega

- [ ] `python -m pytest tests/ -v` — todos os testes passam (se houver)
- [ ] Todos os 5 cadernos executam sem erro (Restart & Run All em cada um)
- [ ] `logs/caderno_01_modelos_linguagem.log` existe e tem conteúdo
- [ ] `logs/caderno_05_pipeline.log` mostra 8 consultas processadas
- [ ] `requirements.txt` instala sem erro em venv limpo
- [ ] `README.md` permite que um colega reproduza o projeto
- [ ] `.gitignore` exclui `venv/`, `logs/`, `data/bulas/`, `*.gguf`
- [ ] Nenhum arquivo contém chaves, tokens ou senhas
- [ ] PDF contém todas as 26 seções obrigatórias
- [ ] Nome do PDF segue o padrão: `kevin_rodrigues_...pdf`
- [ ] Código em português, sem siglas, nomes descritivos

---

## 3. Commits Atômicos

### Commit 1: README.md
```
docs: Sprint 6 — README completo: instalação, execução, estrutura, troubleshooting
```

### Commit 2: Relatório PDF
```
docs: Sprint 6 — relatório PDF com 26 seções obrigatórias no padrão do professor
```

### Commit 3: Limpeza final e verificação
```
chore: Sprint 6 — verificação final: .gitignore, restarta e executa todos os cadernos
```

---

## 4. Resumo das 6 Sprints

| Sprint | Entregável | Células | Commits | Rubricas |
|---|---|---|---|---|
| 1 | `c01_modelos_llm.ipynb` | 11 | 4 | Rubrica 1 (5/5) |
| 2 | `c02_engenharia_prompt.ipynb` | 11 | 5 | Rubrica 2 (5/5) |
| 3 | `c03_embeddings_busca.ipynb` | 10 | 4 | Rubrica 3 (5/5) |
| 4 | `c04_inferencia_local.ipynb` | 9 | 3 | Rubrica 4 (5/5) |
| 5 | `c05_pipeline_rag.ipynb` | 14 | 6 | Rubrica 5 (9/11) |
| 6 | `README.md` + `relatorio.pdf` | — | 3 | Rubrica 5 (2/11) |
| **Total** | **5 cadernos + 2 docs** | **55 células** | **25 commits** | **30/30 rubricas** |
