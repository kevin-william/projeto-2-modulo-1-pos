# Fase 9: Relatório PDF e README

**Objetivo:** Produzir os artefatos finais de entrega: README.md com instruções de reprodução e relatório técnico em PDF cobrindo todos os 30 itens da rubrica.

**Dependências:** Todas as fases anteriores (0 a 8) concluídas.

**Rubricas cobertas:** Rubrica 5 — itens de documentação (2 itens) + integração final.

---

## Tarefas

### 9.1 Criar `README.md`

Estrutura completa:

- [ ] **Título:** "Detector de Interações Medicamentosas com LLMs, NER e RAG"
- [ ] **Descrição do problema:** 2 parágrafos contextualizando o desafio de verificar interações medicamentosas
- [ ] **Arquitetura resumida:** diagrama ASCII ou referência ao `PLANO_IMPLEMENTACAO.md`
- [ ] **Requisitos:**
  - Python 3.9+
  - 8 GB RAM (mínimo), 16 GB recomendado
  - GPU NVIDIA com 6 GB VRAM (opcional — funciona em CPU)
  - ~2 GB de disco para modelos
- [ ] **Instalação:**
  ```bash
  git clone <repo>
  cd projeto-2-modulo-1-pos
  python -m venv venv
  source venv/Scripts/activate   # Windows (Git Bash / MSYS)
  # ou: venv\Scripts\activate    # Windows (CMD)
  pip install -r requirements.txt
  ```
  - Se GPU NVIDIA (recomendado):
    ```bash
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```
- [ ] **Preparação dos dados:**
  ```bash
  # Coloque as bulas em data/bulas/fonte1/ e data/bulas/fonte2/
  python scripts/preprocess.py
  python scripts/embeddings.py
  ```
- [ ] **Anotação e fine-tuning (opcional — modelo pré-treinado incluído):**
  ```bash
  python scripts/annotate.py
  python scripts/train_classifier.py
  ```
- [ ] **Execução:**
  ```bash
  jupyter notebook
  # Abrir notebooks na ordem: c01 → c02 → c03 → c04 → c05
  ```
- [ ] **Configuração:**
  ```bash
  cp .env.example .env
  # Editar .env e preencher OPENAI_API_KEY (opcional — apenas para Notebooks 02, 04, 05)
  ```
- [ ] **Estrutura do projeto:** árvore de diretórios
- [ ] **Troubleshooting:**
  - `CUDA not available` → reinstalar PyTorch com CUDA (ver instrução acima)
  - `Out of memory` → reduzir `BATCH_SIZE` em `scripts/config.py`
  - `Modelo não encontrado` → verificar conexão com internet (download do Hugging Face)
  - `OPENAI_API_KEY not set` → modo offline: células OpenAI pulam com aviso
- [ ] **Status das fases:** tabela

| Fase | Descrição | Status |
|---|---|---|
| 0 | Estrutura + Setup GPU | ✅ Concluída |
| 1 | Pré-processamento | ✅ Concluída |
| 2 | Notebook 01 — HF + NLP | ✅ Concluída |
| 3 | Anotação do Dataset | ✅ Concluída |
| 4 | Fine-Tuning BioBERTpt | ✅ Concluída |
| 5 | Notebook 02 — Prompt Engineering | ✅ Concluída |
| 6 | Notebook 03 — Embeddings + ChromaDB | ✅ Concluída |
| 7 | Notebook 04 — Inferência | ✅ Concluída |
| 8 | Notebook 05 — RAG Pipeline | ✅ Concluída |
| 9 | Relatório + README | 🟡 Em andamento |

### 9.2 Criar Relatório PDF

Estrutura conforme exigido pelo professor. Cada seção mapeada para o sumário obrigatório:

- [ ] **1. Identificação**
  - Nome do aluno: Kevin Rodrigues
  - Nome da disciplina: Sistemas Cognitivos com Large Language Models
  - Título do projeto: "Detector de Interações Medicamentosas com LLMs, NER e RAG"

- [ ] **2. Descrição do problema escolhido**
  - Contexto: profissionais de saúde precisam verificar interações rapidamente
  - Dificuldade: bulas são extensas (até 10 mil tokens), linguagem técnica
  - Solução proposta: sistema cognitivo com NER + classificação + RAG

- [ ] **3. Descrição do corpus ou base de conhecimento**
  - 5.960 bulas: 4.978 ANVISA + 982 Consultaremedios
  - Características de cada fonte (remeter ao `RESUMO_DATASET_BULAS.md`)
  - Exemplos de trechos com interações

- [ ] **4. Justificativa para uso de LLMs**
  - NER: impossível com regex (variabilidade de nomes)
  - Classificação: requer compreensão semântica além de palavras-chave
  - RAG: fundamentação em documentos reais, redução de alucinação

- [ ] **5. Modelos, APIs ou ferramentas utilizadas**
  - Tabela completa da stack (ver `PLANO_IMPLEMENTACAO.md` seção 2)
  - Justificativa para cada escolha

- [ ] **6. Tarefas NLP implementadas**
  - NER, classificação de texto, question answering, summarization, text generation, fill-mask
  - Referência ao Notebook 01

- [ ] **7. Estratégia de prompting**
  - 3 técnicas: zero-shot, few-shot (3 exemplos), chain-of-thought
  - Template base com [PAPEL], [TAREFA], [CONTEXTO], [FORMATO]
  - Referência ao Notebook 02

- [ ] **8. Prompts utilizados e versões testadas**
  - Reproduzir os 3 prompts completos
  - Documentar iterações e mudanças entre versões

- [ ] **9. Estratégia de avaliação dos prompts**
  - 200 pares anotados com ground truth
  - Métricas: acurácia, F1 por classe, % JSON válido, latência
  - Tabela comparativa das 3 técnicas

- [ ] **10. Uso de JSON, parsing ou saída estruturada**
  - Função `parse_interaction_response()` com fallback regex
  - Exemplos de saída válida e tratamento de erro
  - Referência ao Notebook 02, seção 5.7

- [ ] **11. Modelos de embeddings utilizados**
  - 3 modelos comparados: BERT pt, E5 multilíngue, MiniLM
  - Métricas: Precision@5, MRR, latência
  - Referência ao Notebook 03

- [ ] **12. Estratégia de busca vetorial, híbrida ou equivalente**
  - Cosseno (ChromaDB) + BM25 (rank-bm25), alpha=0.3
  - Justificativa da escolha
  - Referência ao Notebook 03

- [ ] **13. Exemplos de consultas e documentos recuperados**
  - 10 queries com top-3 resultados
  - 5 acertos + 5 falhas com análise

- [ ] **14. Estratégia de execução local, remota ou privada**
  - GPT4All (Phi-3-mini, local) + OpenAI (GPT-4o-mini, remota)
  - Classe `LLMProvider` com interface unificada
  - Referência ao Notebook 04

- [ ] **15. Justificativa sobre privacidade, custo, latência e controle**
  - Tabela com 5 dimensões comparadas
  - Análise LGPD para dados de saúde
  - Recomendação: API para protótipo, local para produção

- [ ] **16. Descrição do pipeline RAG ou mecanismo equivalente**
  - Diagrama de arquitetura (reproduzir do plano)
  - Fluxo: consulta → NER → busca → classificação → LLM → JSON
  - Referência ao Notebook 05

- [ ] **17. Estratégia de chunking**
  - 3 estratégias comparadas: sentenças, 3 sentenças, 5 sentenças
  - Métricas: recall@5, tokens/prompt
  - Conclusão: sentenças individuais

- [ ] **18. Vector store ou mecanismo de recuperação utilizado**
  - ChromaDB (persistente, cosseno)
  - Justificativa: sem servidor externo, API nativa, metadados

- [ ] **19. Exemplos de consultas e respostas**
  - 8 consultas de demonstração com JSON de saída completo
  - Referência ao Notebook 05, seção 8.5

- [ ] **20. Análise de respostas com e sem contexto recuperado**
  - 10 consultas comparadas: modo A (sem RAG) vs modo B (com RAG)
  - Exemplo de alucinação reduzida
  - Referência ao Notebook 05, seção 8.6

- [ ] **21. Análise de falhas do pipeline**
  - 3 cenários documentados: NER falha, chunk irrelevante, classificador erra
  - Causa, impacto, mitigação para cada um
  - Referência ao Notebook 05, seção 8.9

- [ ] **22. Riscos de segurança identificados**
  - Prompt injection (demonstrado)
  - Vazamento de contexto (avaliado)
  - Data poisoning (discutido)
  - Referência ao Notebook 05, seção 8.8

- [ ] **23. Controles propostos**
  - Sanitização de input
  - Validação de JSON
  - Threshold de confiança
  - Logging de consultas (para auditoria)
  - Isolamento do ChromaDB (local)

- [ ] **24. Instruções de reprodução**
  - Remeter ao README.md
  - Resumo dos comandos principais

- [ ] **25. Limitações da solução**
  - NER treinado em corpus geral, não específico de bulas brasileiras
  - Dataset de fine-tuning pequeno (~1.500 pares)
  - Sem suporte a interações com 3+ medicamentos
  - Cobertura limitada a medicamentos nas 5.960 bulas
  - Classificador não lida com interações medicamento-alimento

- [ ] **26. Melhorias futuras**
  - Fine-tuning do NER com anotações das bulas
  - Expandir dataset com mais bulas
  - Interface web (Streamlit/Gradio)
  - Atualização automática da base
  - Suporte a interações complexas (3+ medicamentos, alimento, exame)

### 9.3 Gerar PDF

- [ ] Compilar relatório em PDF (Word → PDF, LaTeX → PDF, ou Markdown → PDF via Pandoc)
- [ ] Nome do arquivo: `kevin_rodrigues_sistemas-cognitivos-linguagem-natural_aplicacoes-llms.pdf`
- [ ] Verificar: todas as 26 seções presentes, imagens legíveis, formatação consistente

### 9.4 Revisão final

- [ ] Executar `python -m pytest tests/ -v` — todos os testes passam
- [ ] Executar todos os notebooks (Restart & Run All) — sem erros
- [ ] Verificar `.gitignore` — `.env`, `data/bulas/`, `data/chroma_db/`, `data/modelos_finetuned/` excluídos
- [ ] Verificar `.env.example` — não contém chave real
- [ ] Fazer checklist final das 30 rubricas (ver `PLANO_IMPLEMENTACAO.md` Apêndice)

---

## Artefatos Produzidos

```
README.md
kevin_rodrigues_sistemas-cognitivos-linguagem-natural_aplicacoes-llms.pdf
```

---

## Verificação

- [ ] `python -m pytest tests/ -v` — 100% passam
- [ ] Todos os 5 notebooks executam sem erro (Restart & Run All)
- [ ] README.md permite que um colega reproduza o projeto em < 30 min
- [ ] PDF contém todas as 26 seções obrigatórias
- [ ] PDF não contém chaves, tokens ou dados sensíveis
- [ ] Nome do PDF segue o padrão: `kevin_rodrigues_sistemas-cognitivos-linguagem-natural_aplicacoes-llms.pdf`
- [ ] Commit final: `git add . && git commit -m "docs: Fase 9 — relatório final e README — entrega do projeto"`
