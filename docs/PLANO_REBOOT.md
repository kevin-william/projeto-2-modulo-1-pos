# Plano de Reboot — Detector de Interações Medicamentosas

**Versão:** 3.1 (GPT4All local + logging estruturado)  
**Projeto:** Módulo 1 — Sistemas Cognitivos com Large Language Models  
**Aluno:** Kevin Rodrigues  

---

## Sumário

1. [Diagnóstico do Projeto Atual](#1-diagnóstico-do-projeto-atual)
2. [Princípios do Reboot](#2-princípios-do-reboot)
3. [Nova Arquitetura](#3-nova-arquitetura)
4. [Estrutura de Arquivos](#4-estrutura-de-arquivos)
5. [Fluxo Completo do Projeto](#5-fluxo-completo-do-projeto)
6. [Notebook 01 — Modelos LLM e NLP](#6-notebook-01--modelos-llm-e-nlp)
7. [Notebook 02 — Prompt Engineering](#7-notebook-02--prompt-engineering)
8. [Notebook 03 — Embeddings e Busca Vetorial](#8-notebook-03--embeddings-e-busca-vetorial)
9. [Notebook 04 — Inferência Local](#9-notebook-04--inferência-local)
10. [Notebook 05 — Pipeline RAG](#10-notebook-05--pipeline-rag)
11. [README + Relatório PDF](#11-readme--relatório-pdf)
12. [Mapeamento Rubricas → Células](#12-mapeamento-rubricas--células)
13. [Plano de Execução](#13-plano-de-execução)

---

## 1. Diagnóstico do Projeto Atual

### 1.1 O que foi construído (v2.0)

O projeto atual contém 13 commits, ~20 arquivos de código/doc, e uma pipeline
de dados que processa 5.960 bulas em 270.608 chunks:

| Camada | Arquivos | Linhas | Propósito original | Problema |
|---|---|---|---|---|
| `scripts/config.py` | 1 | 116 | Constantes globais | 90% das constantes são usadas por 1-2 scripts |
| `scripts/preprocess.py` | 1 | 330 | Extrair seções, chunk, JSONL | Só os notebooks 03 e 05 precisam disso |
| `scripts/annotate.py` | 1 | 399 | Weak supervision + curadoria | Over-engineered: 4 funções de exportação, balanceamento, CLI |
| `scripts/validate_annotations.py` | 1 | 80 | 4 regras de validação | Só existe porque o annotate.py é complexo demais |
| `_build_nb.py` | 1 | 500+ | Gerar .ipynb via nbformat | Quebrou 3 vezes (transformers 5.x, escaping, sintaxe) |
| `docs/fases/FASE_0.md ... FASE_9.md` | 10 | ~30 KB | Plano detalhado por fase | Nunca foram consultados durante a implementação real |
| `docs/GUIA_ANOTACAO.md` | 1 | 22 KB | Regras de classificação | Pressupõe um pipeline de anotação que não existe |
| `docs/VALIDACAO_ANOTACOES.md` | 1 | 11 KB | Guia para revisor | Mesmo problema |
| `docs/PASSO_A_PASSO_ANOTACAO.md` | 1 | 14 KB | Tutorial para anotador | Mesmo problema |
| `tests/test_preprocess.py` | 1 | 250 | 17 testes unitários | Testa funções que só existem no script |

### 1.2 Exemplo concreto de over-engineering

O fluxo para gerar anotações automáticas no v2.0:

```
1. config.py        → define GRAVE_KEYWORDS, LEVE_KEYWORDS (30 linhas de listas)
2. preprocess.py    → lê 5.960 .txt → extrai seções → chunk → salva JSONL (270k linhas)
3. annotate.py      → lê JSONL → classifica heuristicamente → gera CSV (695 pares)
4. validate_annotations.py → lê CSV → aplica 4 regras → exporta suspeitos
5. planilha Excel   → revisão manual dos suspeitos
6. annotate.py      → re-importa corrigidos → balanceia → train/val/test
```

Isso são **6 etapas, 4 scripts, 3 formatos de arquivo** (JSONL → CSV → CSV corrigido → CSV balanceado).

No reboot, o notebook 02 simplesmente tem um dicionário Python inline com 30
pares de ground truth. Fim. A demonstração de prompt engineering não precisa de
um pipeline de anotação — precisa de 30 exemplos representativos.

### 1.3 O que deu certo e deve ser preservado

| Ativo | Por que manter |
|---|---|
| **Conhecimento do dataset.** Sabemos que a Fonte 1 usa `## SEÇÃO` e a Fonte 2 usa `[P: ...] R: ...`. Sabemos que `clinicalnerpt-chemical` extrai tanto princípios ativos quanto nomes comerciais. | Esse conhecimento é o diferencial do projeto — está embutido nas células Markdown dos notebooks. |
| **Dados pré-processados.** O `python-processador-bulas/data/pruned/` já contém as bulas podadas (apenas seções clínicas). | Copiar essa pasta para `data/bulas/` resolve o problema de dados em 5 minutos. |
| **Modelos certos.** `clinicalnerpt-chemical` para NER, `bert-base-portuguese-cased` para embeddings. | Já testados, já baixados no cache do HuggingFace. |
| **Notebook 01 executando.** 26 células, 6 pipelines, Restart & Run All passou. | O conhecimento de como usar `pipeline()`, `AutoModel` e `AutoTokenizer` está documentado nos outputs. |

### 1.4 Lições aprendidas

1. **Scripts separados criam distância cognitiva.** Quando o aluno implementa
   `preprocess.py` na semana 1 e vai usar os chunks no notebook 05 na semana 4,
   ele já esqueceu o formato exato do JSONL e precisa reler o código.

2. **Documentação abundante ≠ documentação útil.** Os 10 arquivos de fase foram
   escritos antes da implementação e nunca atualizados. O `GUIA_ANOTACAO.md`
   pressupõe um workflow que nunca foi executado.

3. **Build scripts para notebooks são frágeis.** `_build_nb.py` quebrou com
   transformers 5.x (remoção dos pipelines `summarization` e `question-answering`),
   com escaping de strings no bash do Windows, e com sintaxe de f-string dentro
   de string dentro de code cell.

4. **Complexidade é cumulativa.** Cada script adicionado resolve um problema
   imediato mas cria 2-3 novos pontos de falha. O `annotate.py` precisou do
   `validate_annotations.py` que precisou do `PASSO_A_PASSO_ANOTACAO.md`.

---

## 2. Princípios do Reboot

### Princípio 1: Cada notebook é autossuficiente

**Antes (v2.0):**
```python
# c01_modelos_llm.ipynb, célula 2:
from scripts.config import DEVICE, NER_MODEL, EMBEDDING_MODEL
```

**Depois (v3.1):**
```python
# c01_modelos_llm.ipynb, célula 2:
import torch, logging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NER_MODEL = "pucpr/clinicalnerpt-chemical"
EMBEDDING_MODEL = "neuralmind/bert-base-portuguese-cased"
```

### Princípio 2: Dados carregados diretamente do disco

**Antes:** `data/chunks_bulas.jsonl` (270k linhas, 50 MB) gerado por `preprocess.py`.

**Depois:** O notebook que precisa de chunks lê os `.txt` na hora (20 linhas de função).

### Princípio 3: Menos arquivos, mais células

**Antes:** `annotate.py` (399 linhas) exporta 4 CSVs diferentes com 3 funções de balanceamento.

**Depois:** Não existe. O notebook 02 tem 30 pares em um dict inline.

### Princípio 4: Sem build scripts

**Antes:** `_build_nb.py` gera `.ipynb` via `nbformat`.

**Depois:** Abrir Jupyter → escrever código → executar → salvar. Fim.

### Princípio 5: 100% local, zero APIs externas

**Antes:** 15 pacotes no `requirements.txt` incluindo `openai`, `chromadb`, `rank-bm25`,
`nbformat`, `jupyter`, `nbconvert`.

**Depois:** 7 pacotes essenciais. **GPT4All como único LLM**, usando o binding
Python direto (`from gpt4all import GPT4All`). Fallback para API local
(`http://localhost:4891/v1`) se o binding direto falhar. **Nenhuma dependência
de internet** para inferência — todos os modelos rodam localmente.

```requirements.txt
torch>=2.6.0
transformers>=4.40.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
gpt4all>=2.8.0
pandas>=2.0.0
python-dotenv>=1.0.0
```

### Princípio 6: Duplicação consciente > acoplamento

Se 3 notebooks precisam da constante `NER_MODEL = "pucpr/clinicalnerpt-chemical"`,
ela aparece 3 vezes.

### Princípio 7: Logging estruturado em todos os notebooks

**TODOS** os notebooks seguem o mesmo padrão de logging:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/notebook_XX.log"),  # XX = 01 a 05
        logging.StreamHandler()  # também imprime no notebook
    ]
)
log = logging.getLogger(__name__)
```

**O que é logado:**
- **Setup:** carregamento de cada modelo, device (GPU/CPU), VRAM disponível
- **Processamento:** início/fim de cada etapa (NER, embeddings, busca, classificação)
- **Dados:** quantidade de chunks carregados, tamanho do índice FAISS
- **Consultas:** cada consulta do usuário, medicamentos extraídos, chunks recuperados,
  classe atribuída, tempo total
- **Erros:** qualquer exceção com traceback completo
- **Métricas:** acurácia, F1, latência (quando aplicável)

**Por que logging e não print()?**
- Timestamps automáticos permitem medir latência de cada etapa
- Arquivo de log persiste após fechar o notebook (auditoria)
- Níveis (INFO/WARNING/ERROR) permitem filtrar por severidade
- O professor pode ver o fluxo completo de execução sem precisar rolar células

---

## 3. Nova Arquitetura

### 3.1 Diagrama de fluxo completo

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ETAPA 0: DADOS                                │
│                                                                       │
│  data/bulas/fonte1/*.txt  (4.978 bulas ANVISA — texto corrido)       │
│  data/bulas/fonte2/*.txt  (982 bulas Consultaremedios — Q&A)         │
│                                                                       │
│  Formato Fonte 1:                                                     │
│    ## INTERAÇÕES MEDICAMENTOSAS                                       │
│    A probenecida reduz a secreção tubular renal da amoxicilina...    │
│                                                                       │
│  Formato Fonte 2:                                                     │
│    [P: INTERAÇÃO MEDICAMENTOSA?]                                      │
│    R: Miopatia pode ocorrer em pacientes que usam Zarator...          │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           │  (Carregados sob demanda pelos notebooks)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ETAPA 1: NER (Notebook 05)                         │
│                                                                       │
│  Modelo: pucpr/clinicalnerpt-chemical (BERT 110M, GPU)               │
│  Input:  "Posso tomar Amoxicilina com Ibuprofeno?"                   │
│  Output: ["amoxicilina", "ibuprofeno"]                                │
│                                                                       │
│  Log: "NER extraiu 2 entidades: amoxicilina, ibuprofeno"             │
│                                                                       │
│  Tratamento de erro:                                                  │
│    - Se 0 entidades → "Não identifiquei medicamentos"                 │
│    - Se 1 entidade → "Especifique pelo menos 2 medicamentos"          │
│    - Se 3+ entidades → gera todos os pares (combinação 2 a 2)        │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 ETAPA 2: BUSCA VETORIAL (Notebooks 03, 05)            │
│                                                                       │
│  Embeddings: neuralmind/bert-base-portuguese-cased (768 dims)         │
│  Index:     FAISS IndexFlatIP (inner product = cosseno normalizado)   │
│                                                                       │
│  Log: "Busca FAISS: query='amoxicilina ibuprofeno', top-5 scores=[...]"│
│                                                                       │
│  Fluxo:                                                               │
│  1. Para cada medicamento_outro, gera embedding da query              │
│  2. Busca top-10 chunks no FAISS                                      │
│  3. Filtra chunks que mencionam AMBOS os medicamentos                 │
│  4. Se < 2 chunks → expande busca com query alternativa               │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│           ETAPA 3: CLASSIFICAÇÃO + GERAÇÃO (Notebooks 02, 05)         │
│                                                                       │
│  LLM: GPT4All (binding Python direto)                                 │
│  Modelo: Meta-Llama-3-8B-Instruct.Q4_0.gguf (~4.5 GB RAM)            │
│                                                                       │
│  Estratégia de fallback:                                              │
│  1. Tenta GPT4All binding direto (from gpt4all import GPT4All)        │
│  2. Se falhar (modelo não encontrado, OOM):                           │
│     → Tenta GPT4All API server (http://localhost:4891/v1)             │
│  3. Se ambos falharem:                                                │
│     → Classificação heurística por palavras-chave (fallback final)    │
│                                                                       │
│  Prompt (template Few-shot com 3 exemplos):                           │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ [PAPEL] Farmacêutico clínico especializado                       │ │
│  │ [TAREFA] Classificar interação com base nos chunks               │ │
│  │ [CLASSES] 0=SEM, 1=LEVE, 2=GRAVE                                │ │
│  │ [CHUNKS RECUPERADOS] (top-3 do FAISS)                            │ │
│  │ [EXEMPLOS] 1 de cada classe                                      │ │
│  │ [FORMATO] JSON: {"classe": int, "justificativa": str,            │ │
│  │                 "evidencia": str, "fonte": str}                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Log: "LLM classificação: modelo=Llama-3-8B, tempo=3.2s, classe=1"  │
│                                                                       │
│  Parsing da resposta:                                                 │
│  1. Tenta json.loads()                                                │
│  2. Fallback: regex para extrair "classe": (\d)                       │
│  3. Se ambos falham: retorna erro estruturado                         │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      ETAPA 4: SAÍDA                                   │
│                                                                       │
│  JSON estruturado + log consolidado:                                  │
│  {                                                                    │
│    "consulta": "Posso tomar Amoxicilina com Ibuprofeno?",            │
│    "medicamentos_encontrados": ["amoxicilina", "ibuprofeno"],         │
│    "interacoes": [                                                    │
│      {                                                                │
│        "medicamento_alvo": "amoxicilina",                             │
│        "medicamento_outro": "ibuprofeno",                             │
│        "classe": 1,                                                   │
│        "classe_nome": "LEVE_MODERADA",                                │
│        "justificativa": "...",                                        │
│        "evidencia": "...",                                            │
│        "fonte": "105830895_amoxicilina_profissional.txt",             │
│        "confianca": 0.87                                              │
│      }                                                                │
│    ],                                                                 │
│    "log": ["NER: 2 entidades", "FAISS: 5 chunks",                    │
│            "LLM: classe 1, 3.2s", "Total: 4.1s"]                     │
│  }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Tratamento de erros em cada etapa

| Etapa | Erro possível | Tratamento | Log |
|---|---|---|---|
| NER | Nenhuma entidade encontrada | Retorna `{"erro": "..."}` | `WARNING: NER não encontrou entidades` |
| NER | Apenas 1 entidade | Retorna `{"erro": "..."}` | `WARNING: NER encontrou apenas 1 entidade` |
| Busca | Nenhum chunk relevante | Retorna `{"erro": "..."}` | `WARNING: Nenhum chunk relevante para {query}` |
| GPT4All | Modelo .gguf não encontrado | Tenta download automático | `INFO: Baixando modelo {nome}...` |
| GPT4All | OOM (memória insuficiente) | Fallback para API server | `WARNING: OOM no binding direto, tentando API server` |
| GPT4All | API server offline | Fallback heurístico | `ERROR: Ambos backends indisponíveis, usando heurística` |
| LLM | JSON inválido | Regex fallback | `WARNING: JSON inválido, usando regex fallback` |
| LLM | Regex também falha | Retorna erro | `ERROR: Falha ao parsear resposta do LLM` |

### 3.3 Stack Final (100% local)

| Componente | Tecnologia | Dispositivo |
|---|---|---|
| NER | `pucpr/clinicalnerpt-chemical` (BERT, 110M) | GPU |
| Embeddings | `neuralmind/bert-base-portuguese-cased` (768d) | GPU sob demanda |
| Vector Store | FAISS IndexFlatIP | RAM |
| LLM Primário | GPT4All binding direto (Llama-3-8B Q4_0) | CPU (~4.5 GB RAM) |
| LLM Fallback | GPT4All API server (localhost:4891) | CPU (app desktop) |
| LLM Fallback 2 | Heurística de palavras-chave | CPU (instantâneo) |
| Interface | Jupyter Notebooks | — |
| Logging | `logging` stdlib → arquivo + stream | Disco |

---

## 4. Estrutura de Arquivos

### 4.1 Árvore completa

```
C:\workspace\python\projeto-2-modulo-1-pos\
│
├── c01_modelos_llm.ipynb              ← 11 células. HF pipelines + logging.
├── c02_prompting.ipynb                ← 10 células. GPT4All, 3 técnicas, logging.
├── c03_embeddings_busca.ipynb         ← 9 células. FAISS, busca híbrida, logging.
├── c04_inferencia_local.ipynb         ← 8 células. GPT4All direct vs API, logging.
├── c05_rag_pipeline.ipynb             ← 13 células. RAG completo, logging.
├── README.md
├── requirements.txt                   ← 7 pacotes (sem openai, sem chromadb)
├── .gitignore                         ← logs/, data/bulas/, *.gguf, __pycache__/
├── data/
│   └── bulas/                         ← Copiado de python-processador-bulas/
│       ├── fonte1/                    ← 4.978 .txt
│       └── fonte2/                    ← 982 .txt
├── logs/                              ← Arquivos de log gerados pelos notebooks
│   ├── notebook_01.log
│   ├── notebook_02.log
│   ├── notebook_03.log
│   ├── notebook_04.log
│   └── notebook_05.log
└── docs/
    └── PLANO_REBOOT.md                ← Este arquivo
```

### 4.2 O que foi removido

| Arquivo removido | Motivo |
|---|---|
| `.env.example` | Sem API keys — GPT4All é 100% local |
| `openai` do requirements.txt | Substituído por GPT4All |
| `chromadb` do requirements.txt | Substituído por FAISS |
| Todos os `scripts/*.py` | Lógica inline nos notebooks |
| Todos os `docs/fases/*.md` | Consolidados neste plano |
| `data/chunks_bulas.jsonl` | Leitura direta dos .txt |
| `data/anotacoes/*.csv` | Ground truth inline no notebook 02 |

---

## 5. Fluxo Completo do Projeto

### 5.1 Setup do ambiente (5 minutos, zero APIs externas)

```bash
cd C:\workspace\python\projeto-2-modulo-1-pos
python -m venv venv
source venv/Scripts/activate

# PyTorch com CUDA (GPU NVIDIA)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Demais dependências (tudo local)
pip install -r requirements.txt

# Copiar dados do projeto de referência
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte1 data/bulas/
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte2 data/bulas/

# Criar diretório de logs
mkdir logs

# O GPT4All fará download automático do modelo na primeira execução
```

### 5.2 Ordem de execução para o professor

```
c01 (HF + NLP)         → "Sei usar transformers, pipelines, AutoModel"
c02 (Prompting)        → "Sei estruturar prompts, 3 técnicas, validar JSON"
c03 (Embeddings)       → "Sei gerar embeddings, FAISS, busca híbrida"
c04 (Inferência Local) → "Sei comparar GPT4All direct vs API server"
c05 (RAG Pipeline)     → "Sei integrar tudo: NER → FAISS → GPT4All → JSON"
```

**Nenhum notebook importa de outro.** Cada um é autossuficiente e gera seu
próprio arquivo de log em `logs/notebook_XX.log`.

### 5.3 O que o professor vê em cada notebook (com logging)

Cada notebook começa com a mesma célula de setup de logging:

```python
import logging
from pathlib import Path
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/notebook_0X.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 50)
log.info("Iniciando Notebook 0X — <título>")
log.info(f"PyTorch: {torch.__version__} | Device: {DEVICE}")
```

Ao final de cada notebook, a célula de conclusão loga um resumo:

```python
log.info("Notebook 0X concluído com sucesso")
log.info(f"Total de células executadas: {len(In['cells'])}")
```

---

## 6. Notebook 01 — Modelos LLM e NLP

**Rubrica 1 (5 itens):** demonstrar modelos pré-treinados, configurar tokenizers,
comparar arquiteturas, explicar diferenças, relacionar ao domínio.

### Células (11 — 1 a mais para logging)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | **Setup + logging:** `import torch, logging`, configurar `logs/notebook_01.log` |
| 3 | Markdown | **2.1 AutoModel + AutoTokenizer** — explicação do estilo do professor |
| 4 | Code | Carregar `clinicalnerpt-chemical`, tokenizar, inspecionar `last_hidden_state.shape`. Log: `log.info("Modelo carregado: %s", model_id)` |
| 5 | Markdown | **2.2 sentiment-analysis em frases clínicas** |
| 6 | Code | `pipeline("sentiment-analysis")` com 5 frases de bulas. Log: cada frase classificada. |
| 7 | Markdown | **2.3 NER com clinicalnerpt-chemical** |
| 8 | Code | NER em trecho real de bula. Log: `log.info("NER extraiu %d entidades: %s", len(unicos), unicos)` |
| 9 | Markdown | **2.4 Tabela comparativa de arquiteturas** |
| 10 | Code + Markdown | Tabela: encoder-only, decoder-only, encoder-decoder. Pipeline vs manual. |
| 11 | Markdown | Conclusão: quais tarefas importam + `log.info("Notebook 01 concluído")` |

**Logs gerados (exemplo `notebook_01.log`):**
```
2026-06-21 10:00:01 [INFO] ==================================================
2026-06-21 10:00:01 [INFO] Iniciando Notebook 01 — Modelos LLM e NLP
2026-06-21 10:00:01 [INFO] PyTorch: 2.6.0+cu124 | Device: cuda
2026-06-21 10:00:05 [INFO] Modelo carregado: pucpr/clinicalnerpt-chemical
2026-06-21 10:00:05 [INFO] Output shape: torch.Size([1, 8, 768])
2026-06-21 10:00:05 [INFO] Tokens WordPiece: 8 tokens
2026-06-21 10:00:08 [INFO] Sentiment: 5 frases classificadas
2026-06-21 10:00:12 [INFO] NER extraiu 4 entidades: ['acenocumarol', 'alopurinol', 'amoxicilina', 'varfarina']
2026-06-21 10:00:12 [INFO] Notebook 01 concluído com sucesso
```

---

## 7. Notebook 02 — Prompt Engineering

**Rubrica 2 (5 itens):** chamadas a modelos, 3+ técnicas, prompts estruturados,
saída JSON com parsing, avaliação e iteração.

### 7.1 Estratégia de LLM (GPT4All com fallback)

O notebook 02 implementa uma classe `LLMProvider` que encapsula a estratégia
de fallback em 3 níveis:

```python
class LLMProvider:
    """
    Provedor de LLM com fallback em 3 níveis:

    Nível 1: GPT4All binding Python direto (from gpt4all import GPT4All)
             → modelo .gguf carregado em memória, inferência CPU
    Nível 2: GPT4All API server (http://localhost:4891/v1)
             → requer app desktop com API enabled
    Nível 3: Heurística de palavras-chave
             → classificação instantânea, sem LLM
    """

    def __init__(self, model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
        self.model_name = model_name
        self.model = None
        self.backend = None  # "direct", "api", "heuristic"
        self._init_backend()

    def _init_backend(self):
        # Nível 1: binding direto
        try:
            from gpt4all import GPT4All
            self.model = GPT4All(self.model_name)
            self.backend = "direct"
            log.info("GPT4All backend: direct binding (%s)", self.model_name)
            return
        except Exception as e:
            log.warning("GPT4All direct binding falhou: %s", e)

        # Nível 2: API server
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="http://localhost:4891/v1",
                api_key="gpt4all"
            )
            self.client.models.list()  # testa conexão
            self.backend = "api"
            log.info("GPT4All backend: API server (localhost:4891)")
            return
        except Exception as e:
            log.warning("GPT4All API server falhou: %s", e)

        # Nível 3: heurística
        self.backend = "heuristic"
        log.warning("Usando fallback heurístico (sem LLM)")

    def generate(self, prompt, max_tokens=150):
        log.info("LLM generate: backend=%s, prompt_len=%d", self.backend, len(prompt))
        t0 = time.time()

        if self.backend == "direct":
            response = self.model.generate(prompt, max_tokens=max_tokens)
        elif self.backend == "api":
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            response = resp.choices[0].message.content
        else:
            response = self._classificar_heuristicamente(prompt)

        elapsed = time.time() - t0
        log.info("LLM response: backend=%s, time=%.2fs, len=%d",
                 self.backend, elapsed, len(response))
        return response
```

### 7.2 Ground truth (30 pares)

Mesmos 30 pares balanceados (10 por classe) da versão anterior do plano,
extraídos das bulas reais.

### 7.3 Células (10)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | **Setup + logging:** `logs/notebook_02.log`. Classe `LLMProvider` (com fallback 3 níveis). Carregar 30 pares ground truth. `log.info("Ground truth: %d pares", len(GROUND_TRUTH))` |
| 3 | Markdown | **Template de prompt base:** papel + tarefa + classes + contexto + formato JSON |
| 4 | Code + Markdown | **Técnica 1 — Zero-shot.** `log.info("Iniciando Zero-shot: %d pares", len(pares))`. Métricas. |
| 5 | Code + Markdown | **Técnica 2 — Few-shot (3 exemplos).** Métricas. |
| 6 | Code + Markdown | **Técnica 3 — Chain-of-Thought.** Instrução "pense passo a passo". Métricas. |
| 7 | Code | `parse_interaction_response()` — 3 estratégias de fallback. |
| 8 | Markdown | **Tabela comparativa** com acurácia, F1, JSON válido, latência, backend usado. |
| 9 | Markdown | Prompt injection: demonstração de ataque + sanitização. |
| 10 | Markdown | Conclusão + `log.info("Notebook 02 concluído. Backend: %s", llm.backend)` |

### 7.4 Logs gerados (exemplo)

```
2026-06-21 11:00:01 [INFO] ==================================================
2026-06-21 11:00:01 [INFO] Iniciando Notebook 02 — Prompt Engineering
2026-06-21 11:00:05 [INFO] GPT4All backend: direct binding (Meta-Llama-3-8B-Instruct.Q4_0.gguf)
2026-06-21 11:00:05 [INFO] Ground truth: 30 pares (0: 10, 1: 10, 2: 10)
2026-06-21 11:00:05 [INFO] Iniciando Zero-shot: 30 pares
2026-06-21 11:00:05 [INFO] LLM generate: backend=direct, prompt_len=450
2026-06-21 11:00:08 [INFO] LLM response: backend=direct, time=3.20s, len=85
2026-06-21 11:01:45 [INFO] Zero-shot concluído: acurácia=0.65, JSON válido=0.85
2026-06-21 11:01:45 [INFO] Iniciando Few-shot: 30 pares
...
2026-06-21 11:05:30 [INFO] Few-shot concluído: acurácia=0.82, JSON válido=0.95
2026-06-21 11:05:30 [INFO] Iniciando Chain-of-Thought: 30 pares
...
2026-06-21 11:10:15 [INFO] CoT concluído: acurácia=0.78, JSON válido=0.90
2026-06-21 11:10:15 [INFO] Notebook 02 concluído. Backend: direct
```

---

## 8. Notebook 03 — Embeddings e Busca Vetorial

**Rubrica 3 (5 itens):** gerar embeddings, busca semântica/híbrida, avaliar modelos,
analisar acertos/falhas, justificar estratégia.

### Células (9 — 1 a mais para logging)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica |
| 2 | Code | **Setup + logging:** `logs/notebook_03.log`. `SentenceTransformer`, `faiss`. `log.info("Modelo embeddings: %s", EMBEDDING_MODEL)` |
| 3 | Code | Função `carregar_chunks()` — lê .txt, extrai seções, chunk. `log.info("Chunks carregados: %d (F1: %d, F2: %d)")` |
| 4 | Code | Gerar embeddings, FAISS. `log.info("Embeddings: %d vetores, %d dims", N, d)`. `log.info("FAISS indexado: %d vetores", index.ntotal)` |
| 5 | Code + Markdown | **Busca semântica (cosseno).** 10 queries. `log.info("Busca: '%s' → top-5 scores: %s")` |
| 6 | Code + Markdown | **Busca híbrida (BM25 + embeddings).** Comparação. |
| 7 | Code + Markdown | **Comparação de 2 modelos:** BERT pt vs MiniLM. Precision@5. |
| 8 | Markdown | Análise de 3 acertos + 3 falhas. Justificativa. |
| 9 | Markdown | `log.info("Notebook 03 concluído. FAISS: %d vetores", index.ntotal)` |

---

## 9. Notebook 04 — Inferência Local

**Rubrica 4 (5 itens):** executar modelos locais, comparar backends, integrar
programaticamente, analisar trade-offs, considerar privacidade.

### 9.1 Estratégia

O notebook 04 **compara os 3 backends** disponíveis no GPT4All:

1. **Binding direto** (`from gpt4all import GPT4All`) — modelo carregado em memória
2. **API server** (`http://localhost:4891/v1`) — app desktop como servidor
3. **Heurística** (fallback sem LLM) — classificação por palavras-chave

**Dimensões comparadas:** qualidade (acurácia/F1), latência, consumo de RAM,
facilidade de setup, disponibilidade offline.

### 9.2 Células (8)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica. Explicação dos 3 backends. |
| 2 | Code | **Setup + logging:** `logs/notebook_04.log`. Instanciar os 3 backends: `GPT4AllDirect`, `GPT4AllAPI`, `HeuristicClassifier`. Log do status de cada um. |
| 3 | Code + Markdown | **Qualidade:** classificar 30 pares com cada backend. `log.info("Backend %s: acurácia=%.2f, F1=%.2f")` |
| 4 | Code + Markdown | **Latência:** 10 consultas, média/p95. `log.info("Backend %s: latência média=%.0fms, P95=%.0fms")` |
| 5 | Code + Markdown | **Consumo de RAM:** `psutil.Process().memory_info().rss` antes/depois de carregar cada backend. |
| 6 | Markdown | **Setup e disponibilidade:** binding direto (download automático), API server (requer app desktop), heurística (instantâneo). |
| 7 | Markdown | **Privacidade:** todos os backends são 100% locais — dados nunca saem da máquina. Comparação com APIs cloud (OpenAI) em tabela teórica. |
| 8 | Markdown | **Conclusão:** tabela 5 dimensões + recomendação. `log.info("Notebook 04 concluído")` |

### 9.3 Tabela comparativa

| Dimensão | Binding Direto | API Server | Heurística |
|---|---|---|---|
| Qualidade (F1) | ~0.78 | ~0.78 | ~0.55 |
| Latência | ~3s | ~1s | <1ms |
| RAM | ~5 GB | 0 (app externo) | 0 |
| Setup | `pip install gpt4all` | App desktop + enable API | Nada |
| Offline | ✅ | ✅ | ✅ |
| Privacidade | ✅ 100% local | ✅ 100% local | ✅ 100% local |

---

## 10. Notebook 05 — Pipeline RAG

**Rubrica 5 (11 itens):** pipeline completo, vector store, chunking strategies,
com/sem contexto, falhas, segurança, problema aderente, executável, integrado,
decisões justificadas, sem expor chaves, análise crítica.

### 10.1 Células (13 — 1 a mais para logging)

| # | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica, diagrama ASCII da arquitetura |
| 2 | Code | **Setup + logging:** `logs/notebook_05.log`. Carregar NER, embeddings, LLMProvider (GPT4All). `log.info("Modelos carregados: NER=%s, EMB=%s, LLM=%s")` |
| 3 | Code | **Carregar e indexar bulas:** `carregar_e_indexar()`. `log.info("Chunks: %d, FAISS: %d vetores")` |
| 4 | Code | **Função `consultar(query)`:** NER → FAISS → prompt → GPT4All → parse JSON. Logging em CADA etapa. |
| 5 | Code + Markdown | **8 consultas demo.** `log.info("Consulta: '%s' → %d interações encontradas")` |
| 6 | Code + Markdown | **Comparação com vs sem contexto.** |
| 7 | Code + Markdown | **Chunking strategies.** |
| 8 | Markdown | **Análise de falhas:** 3 cenários. |
| 9 | Code + Markdown | **Prompt injection:** demo + sanitização. |
| 10 | Markdown | **Riscos de segurança.** |
| 11 | Markdown | **Limitações.** |
| 12 | Markdown | **Conclusão.** |
| 13 | Code | `log.info("Notebook 05 concluído. Total consultas: %d", total)` |

### 10.2 Função `consultar()` com logging detalhado

```python
def consultar(query):
    t0 = time.time()
    log.info("=" * 40)
    log.info("Consulta: '%s'", query)

    # ETAPA 1: NER
    log.info("Etapa 1/4: NER...")
    entidades_raw = ner(query)
    medicamentos = list(set(e["word"].lower() for e in entidades_raw))
    log.info("  Entidades extraídas: %s", medicamentos)

    if len(medicamentos) < 2:
        log.warning("  Apenas %d entidade(s) — impossível formar par", len(medicamentos))
        return {"erro": "Especifique pelo menos 2 medicamentos",
                "medicamentos_encontrados": medicamentos}

    # ETAPA 2: Busca FAISS
    interacoes = []
    for i in range(len(medicamentos)):
        for j in range(i+1, len(medicamentos)):
            alvo, outro = medicamentos[i], medicamentos[j]
            log.info("Etapa 2/4: Busca FAISS para '%s' + '%s'", alvo, outro)

            q = f"{alvo} {outro} interação"
            q_emb = embedder.encode([q], normalize_embeddings=True).astype(np.float32)
            dists, idxs = index.search(q_emb, 10)

            relevantes = []
            for dist, idx in zip(dists[0], idxs[0]):
                c = chunks[idx]
                if alvo in c["texto"].lower() and outro in c["texto"].lower():
                    relevantes.append({"chunk": c, "score": float(dist)})
                if len(relevantes) >= 3:
                    break

            log.info("  Chunks relevantes: %d (top score: %.3f)",
                     len(relevantes),
                     relevantes[0]["score"] if relevantes else 0)

            if not relevantes:
                log.warning("  Nenhum chunk relevante encontrado")
                continue

            # ETAPA 3: GPT4All
            log.info("Etapa 3/4: Classificação via GPT4All...")
            chunks_txt = "\n\n".join(
                f"[{r['chunk']['arquivo']}]\n{r['chunk']['texto']}"
                for r in relevantes[:3]
            )
            prompt = build_prompt(alvo, outro, chunks_txt)
            log.info("  Prompt: %d caracteres", len(prompt))

            raw = llm.generate(prompt, max_tokens=200)
            parsed = parse_interaction_response(raw)

            classe = parsed["classe"] if parsed else -1
            log.info("  Resultado: classe=%d (%s)",
                     classe,
                     {0: "SEM", 1: "LEVE", 2: "GRAVE"}.get(classe, "ERRO"))

            interacoes.append({
                "medicamento_alvo": alvo,
                "medicamento_outro": outro,
                "classe": classe,
                "classe_nome": {0: "SEM_INTERACAO", 1: "LEVE_MODERADA",
                                2: "GRAVE_CONTRAINDICADA"}.get(classe, "ERRO"),
                "justificativa": parsed.get("justificativa", "") if parsed else "",
                "evidencia": parsed.get("evidencia", "") if parsed else "",
                "fonte": relevantes[0]["chunk"]["arquivo"] if relevantes else "",
                "confianca": relevantes[0]["score"] if relevantes else 0.0,
            })

    # ETAPA 4: Resultado
    elapsed = (time.time() - t0) * 1000
    log.info("Etapa 4/4: Concluído em %.0fms. %d interações encontradas",
             elapsed, len(interacoes))

    return {
        "consulta": query,
        "medicamentos_encontrados": medicamentos,
        "interacoes": interacoes,
        "tempo_total_ms": elapsed,
    }
```

### 10.3 Logs gerados (exemplo `notebook_05.log`)

```
2026-06-21 15:00:01 [INFO] ==================================================
2026-06-21 15:00:01 [INFO] Iniciando Notebook 05 — Pipeline RAG
2026-06-21 15:00:05 [INFO] Modelos carregados: NER=clinicalnerpt-chemical, EMB=bert-base-portuguese-cased, LLM=Llama-3-8B
2026-06-21 15:00:15 [INFO] Chunks: 5234, FAISS: 5234 vetores
2026-06-21 15:00:15 [INFO] ========================================
2026-06-21 15:00:15 [INFO] Consulta: 'Posso tomar Amoxicilina com Ibuprofeno?'
2026-06-21 15:00:15 [INFO] Etapa 1/4: NER...
2026-06-21 15:00:15 [INFO]   Entidades extraídas: ['amoxicilina', 'ibuprofeno']
2026-06-21 15:00:15 [INFO] Etapa 2/4: Busca FAISS para 'amoxicilina' + 'ibuprofeno'
2026-06-21 15:00:16 [INFO]   Chunks relevantes: 3 (top score: 0.87)
2026-06-21 15:00:16 [INFO] Etapa 3/4: Classificação via GPT4All...
2026-06-21 15:00:19 [INFO]   Resultado: classe=1 (LEVE)
2026-06-21 15:00:19 [INFO] Etapa 4/4: Concluído em 4200ms. 1 interações encontradas
2026-06-21 15:00:19 [INFO] ========================================
2026-06-21 15:00:19 [INFO] Consulta: 'Dipirona e AAS juntos têm problema?'
...
2026-06-21 15:05:30 [INFO] Notebook 05 concluído. Total consultas: 8
```

---

## 11. README + Relatório PDF

### 11.1 README.md

```markdown
# Detector de Interações Medicamentosas com LLMs e RAG

Sistema cognitivo 100% local que recebe consultas em linguagem natural sobre
interações medicamentosas e retorna classificação fundamentada em bulas reais.

**Nenhuma API externa necessária.** Todos os modelos rodam localmente.

## Requisitos

- Python 3.9+
- 8 GB RAM (16 GB recomendado para GPT4All)
- GPU NVIDIA com 6 GB VRAM (opcional — funciona em CPU)
- 6 GB de disco (modelos .gguf + embeddings)

## Instalação

```bash
python -m venv venv
source venv/Scripts/activate

# PyTorch com CUDA (se tiver GPU NVIDIA)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Dados

```bash
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte1 data/bulas/
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte2 data/bulas/
mkdir logs
```

## Execução

Abra os notebooks na ordem:

1. `c01_modelos_llm.ipynb` — Hugging Face, pipelines, AutoModel
2. `c02_prompting.ipynb` — GPT4All, 3 técnicas, JSON parsing
3. `c03_embeddings_busca.ipynb` — Embeddings, FAISS, busca híbrida
4. `c04_inferencia_local.ipynb` — GPT4All direct vs API vs heurística
5. `c05_rag_pipeline.ipynb` — Pipeline RAG completo

Cada notebook gera logs em `logs/notebook_XX.log`.

## Logs

Todos os notebooks geram logs com timestamps. Para acompanhar em tempo real:

```bash
tail -f logs/notebook_05.log
```
```

### 11.2 Relatório PDF

As 26 seções obrigatórias, cada uma referenciando células específicas dos notebooks
e trechos dos arquivos de log como evidência de execução.

---

## 12. Mapeamento Rubricas → Células

### Rubrica 1 (5 itens) → Notebook 01

| # | Item | Onde verificar |
|---|---|---|
| 1.1 | Tarefas NLP com modelos pré-treinados | Célula 6 (sentiment), Célula 8 (NER) — outputs visíveis |
| 1.2 | Configurou tokenizers, pipelines, parâmetros | Célula 4 (AutoModel), Células 6, 8 |
| 1.3 | Comparou modelos/arquiteturas | Célula 10 — tabela 5 modelos, 3 arquiteturas |
| 1.4 | Explicou diferenças (encoder-only vs decoder-only) | Célula 10 (Markdown) |
| 1.5 | Relacionou resultados ao domínio | Célula 11 (conclusão) |

### Rubrica 2 (5 itens) → Notebook 02

| # | Item | Onde verificar |
|---|---|---|
| 2.1 | Chamadas a modelos de linguagem | Células 4, 5, 6 — `llm.generate()` via GPT4All |
| 2.2 | 3 técnicas comparadas | Zero-shot (cél 4), Few-shot (cél 5), CoT (cél 6) |
| 2.3 | Prompts estruturados | Célula 3 — [PAPEL]+[TAREFA]+[CLASSES]+[CONTEXTO]+[FORMATO] |
| 2.4 | JSON + parsing/validação | Célula 7 — `parse_interaction_response()` com 3 estratégias |
| 2.5 | Avaliou e iterou prompts | Célula 8 — tabela comparativa + 3 iterações documentadas |

### Rubrica 3 (5 itens) → Notebook 03

| # | Item | Onde verificar |
|---|---|---|
| 3.1 | Gerou embeddings | Célula 4 — `model.encode()` + FAISS |
| 3.2 | Busca semântica/híbrida | Célula 5 (cosseno), Célula 6 (BM25 híbrida) |
| 3.3 | Avaliou modelos/métricas | Célula 7 — Precision@5 BERT pt vs MiniLM |
| 3.4 | Analisou acertos e falhas | Célula 8 — 3 acertos + 3 falhas com explicação |
| 3.5 | Justificou estratégia | Célula 8 — FAISS, top-k=5, alpha=0.3 |

### Rubrica 4 (5 itens) → Notebook 04

| # | Item | Onde verificar |
|---|---|---|
| 4.1 | Modelo local executado | Célula 2 — 3 backends GPT4All instanciados |
| 4.2 | Comparou dimensões | Cél 3 (qualidade), Cél 4 (latência), Cél 5 (RAM), Cél 6 (setup) |
| 4.3 | Integração programática | Célula 2 — classe `LLMProvider` com interface unificada |
| 4.4 | Vantagens/limitações | Células 6, 7, 8 |
| 4.5 | Privacidade/custo/latência/controle | Células 5 (RAM), 7 (privacidade), 8 (conclusão) |

### Rubrica 5 (11 itens) → Notebook 05 + README + Relatório

| # | Item | Onde verificar |
|---|---|---|
| 5.1 | Pipeline RAG completo | Células 3, 4, 5 |
| 5.2 | Vector store funcional | Célula 3 — FAISS IndexFlatIP |
| 5.3 | Chunking + com/sem contexto | Célula 6 (com/sem RAG), Célula 7 (3 estratégias) |
| 5.4 | Pontos de falha | Célula 8 — 3 cenários |
| 5.5 | Riscos de segurança | Cél 9 (prompt injection), Cél 10 (tabela riscos) |
| 5.6 | Problema aderente | Célula 1 |
| 5.7 | Solução executável/documentada | README.md + logs |
| 5.8 | Integração coerente | Célula 5 — 8 consultas demo |
| 5.9 | Decisões justificadas | Cél 7, 8, 11 |
| 5.10 | Não expôs chaves | Zero APIs externas — nada a expor |
| 5.11 | Análise crítica | Célula 11 (limitações), Célula 12 (conclusão) |

---

## 13. Plano de Execução

### Pré-requisito (5 minutos)

```bash
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte1 \
      C:/workspace/python/projeto-2-modulo-1-pos/data/bulas/
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte2 \
      C:/workspace/python/projeto-2-modulo-1-pos/data/bulas/
mkdir -p logs
```

### Ordem de implementação

| # | Artefato | Tempo | Depende de |
|---|---|---|---|
| 1 | `requirements.txt`, `.gitignore` | 5 min | nada |
| 2 | `c01_modelos_llm.ipynb` (11 células) | 1-2 h | requisitos instalados |
| 3 | `c02_prompting.ipynb` (10 células) | 1-2 h | GPT4All instalado |
| 4 | `c03_embeddings_busca.ipynb` (9 células) | 1-2 h | dados copiados |
| 5 | `c04_inferencia_local.ipynb` (8 células) | 1 h | GPT4All instalado |
| 6 | `c05_rag_pipeline.ipynb` (13 células) | 2-3 h | todos anteriores |
| 7 | `README.md` | 30 min | projeto completo |
| 8 | Relatório PDF (26 seções) | 2-3 h | projeto completo |

**Total: ~12 horas de trabalho focado.**

### Commits atômicos (7)

```
feat: c01_modelos_llm — 11 celulas, HF pipelines + logging
feat: c02_prompting — 10 celulas, GPT4All 3 backends, 30 pares, parsing JSON
feat: c03_embeddings_busca — 9 celulas, FAISS + busca hibrida + logging
feat: c04_inferencia_local — 8 celulas, GPT4All direct vs API vs heuristica
feat: c05_rag_pipeline — 13 celulas, NER + FAISS + GPT4All + seguranca
docs: README + requisitos + instrucoes de reproducao
docs: relatorio PDF com 26 secoes obrigatorias
```

---

## Apêndice: Comparação Antes vs Depois

| Métrica | v2.0 (atual) | v3.1 (reboot) |
|---|---|---|
| APIs externas | OpenAI (paga) | **Nenhuma** — 100% local |
| LLM | GPT-4o-mini (remoto) | GPT4All Llama-3-8B (local) |
| Backend LLM | 1 (OpenAI) | 3 (direct / API / heurística) |
| Dependências pip | 15 | **7** |
| Logging | Inexistente | **5 arquivos de log** com timestamps |
| Arquivos Python (.py) | 8 | 0 |
| Arquivos de doc | 13 | 1 |
| Células de notebook | 26 (só c01) | **51 (total 5 notebooks)** |
| Vector store | ChromaDB | FAISS |
| Dados intermediários | JSONL 270k + 4 CSVs | 0 |
| Tempo estimado | ~21 dias | ~3 dias |
| Commits | 13 | 7 |
| Cobre 30 rubricas? | Sim | Sim (com mapeamento + logs) |
| Requer internet? | Sim (API OpenAI) | **Não** (tudo local) |
| Requer .env / API keys? | Sim | **Não** |
