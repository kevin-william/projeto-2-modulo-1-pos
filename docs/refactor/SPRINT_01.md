# Sprint 1 — Fundação + Notebook 01 (HuggingFace)

**Objetivo:** Criar a estrutura do projeto, instalar dependências, e implementar
o Notebook 01 demonstrando o ecossistema HuggingFace com pipelines aplicados ao
domínio de bulas médicas.

**Duração:** 3-4 horas  
**Commits:** 4 atômicos  
**Rubricas cobertas:** Rubrica 1 (5 itens)

---

## 1. Estrutura de Diretórios e Arquivos

### 1.1 O que criar

```
C:\workspace\python\projeto-2-modulo-1-pos\
├── requirements.txt
├── .gitignore
├── c01_modelos_llm.ipynb
├── data/
│   └── bulas/
│       ├── fonte1/          ← copiar do python-processador-bulas
│       └── fonte2/          ← copiar do python-processador-bulas
├── logs/                    ← criado automaticamente pelo notebook
└── docs/
    └── refactor/
        └── SPRINT_01.md     ← este arquivo
```

### 1.2 Arquivo `requirements.txt`

```
torch>=2.6.0
transformers>=4.40.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
gpt4all>=2.8.0
pandas>=2.0.0
python-dotenv>=1.0.0
```

### 1.3 Arquivo `.gitignore`

```
venv/
__pycache__/
.ipynb_checkpoints/
data/bulas/
logs/
*.gguf
```

### 1.4 Setup do ambiente

```bash
cd C:\workspace\python\projeto-2-modulo-1-pos

# Criar e ativar ambiente virtual
python -m venv venv
source venv/Scripts/activate

# PyTorch com CUDA (GPU NVIDIA)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Demais dependências
pip install -r requirements.txt

# Copiar dados
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte1 data/bulas/
cp -r C:/workspace/python/python-processador-bulas/data/pruned/fonte2 data/bulas/

# Criar diretório de logs
mkdir logs
```

### 1.5 Verificação do ambiente

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponível: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

**Resultado esperado:**
```
PyTorch: 2.6.0+cu124
CUDA disponível: True
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
VRAM: 6.4 GB
```

---

## 2. Notebook 01 — Célula a Célula

### Convenções de nomenclatura (PORTUGUÊS, SEM SIGLAS)

| ❌ Proibido | ✅ Obrigatório |
|---|---|
| `NER_MODEL` | `modelo_reconhecimento_entidades` |
| `DEVICE` | `dispositivo` |
| `EMBEDDING_MODEL` | `modelo_embeddings` |
| `df`, `df2`, `tmp` | `tabela_pares`, `resultado_classificacao` |
| `x`, `y`, `z` | `indice_medicamento`, `contador_interacoes` |
| `ner`, `ner_pipe` | `reconhecedor_entidades` |
| `resp`, `res` | `resposta_modelo`, `resultado_consulta` |

**Regra de ouro para nomes:** O nome de uma variável, método ou classe deve
contar a história do que ela faz. Exemplo:

```python
# ❌ Ruim — não conta história
r = ner(q)

# ✅ Bom — conta história
entidades_encontradas = reconhecedor_entidades(consulta_usuario)
```

### Célula 1 — Cabeçalho (Markdown)

```markdown
# Caderno 01 — Modelos de Linguagem e Processamento de Linguagem Natural com HuggingFace

**Objetivo:** Demonstrar o uso de modelos pré-treinados do ecossistema HuggingFace
aplicados ao domínio de bulas médicas brasileiras, seguindo o estilo do professor
(`pipeline`, `AutoModel`, `AutoTokenizer`).

**Rubrica 1:** Construir aplicações de Processamento de Linguagem Natural com
Modelos de Linguagem de Grande Escala e ecossistema HuggingFace (5 itens).

### Fluxo deste caderno

1. **AutoModel + AutoTokenizer** — carregar modelo, tokenizar, inspecionar dimensões
2. **Análise de sentimento** — pipeline em frases clínicas, demonstrar limitações
3. **Reconhecimento de entidades nomeadas** — extrair medicamentos de bulas reais
4. **Tabela comparativa** — encoder-only vs decoder-only vs encoder-decoder
5. **Conclusão** — quais tarefas importam para o detector de interações
```

### Célula 2 — Configuração e Registro de Execução (Code)

```python
# ─── Importações ─────────────────────────────────────────────────
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
from transformers import pipeline, AutoModel, AutoTokenizer

# ─── Configuração do dispositivo (GPU/CPU) ───────────────────────
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Constantes do caderno ───────────────────────────────────────
modelo_reconhecimento_entidades = "pucpr/clinicalnerpt-chemical"
modelo_embeddings = "neuralmind/bert-base-portuguese-cased"

# ─── Configuração de registro de execução (logging) ─────────────
diretorio_logs = Path("logs")
diretorio_logs.mkdir(exist_ok=True)

formato_log = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Arquivo de log — persiste após fechar o caderno
arquivo_log = diretorio_logs / "caderno_01_modelos_linguagem.log"
manipulador_arquivo = logging.FileHandler(arquivo_log, encoding="utf-8")
manipulador_arquivo.setFormatter(formato_log)

# Console — exibe no próprio caderno
manipulador_console = logging.StreamHandler(sys.stdout)
manipulador_console.setFormatter(formato_log)

registro = logging.getLogger("caderno_01")
registro.setLevel(logging.INFO)
registro.addHandler(manipulador_arquivo)
registro.addHandler(manipulador_console)

# ─── Registro inicial ────────────────────────────────────────────
registro.info("=" * 70)
registro.info("Caderno 01 — Modelos de Linguagem e Processamento de Linguagem Natural")
registro.info("Início da execução: %s", datetime.now().isoformat())
registro.info("PyTorch versão: %s", torch.__version__)
registro.info("CUDA disponível: %s", torch.cuda.is_available())
registro.info("Dispositivo configurado: %s", dispositivo)

if torch.cuda.is_available():
    propriedades_gpu = torch.cuda.get_device_properties(0)
    registro.info(
        "GPU: %s | VRAM total: %.1f GB",
        torch.cuda.get_device_name(0),
        propriedades_gpu.total_memory / 1e9
    )

# Exibir resumo no caderno
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Dispositivo: {dispositivo}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    memoria_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {memoria_gpu_gb:.1f} GB")
print(f"Logs: {arquivo_log.resolve()}")
```

### Célula 3 — Explicação AutoModel (Markdown)

```markdown
## 2.1 Carregamento de Modelo com AutoModel e AutoTokenizer

Seguindo exatamente o padrão demonstrado em aula pelo professor:

1. `AutoTokenizer.from_pretrained(identificador_modelo)` — carrega o tokenizador
   apropriado para o modelo (WordPiece para BERT, BPE para GPT-2, etc.)
2. `AutoModel.from_pretrained(identificador_modelo)` — carrega o corpo do modelo
   **sem cabeçalho de tarefa** (útil para compreender a arquitetura)
3. `modelo(**entradas)` — propagação direta (forward pass), retorna
   `last_hidden_state`

**Modelo escolhido:** `pucpr/clinicalnerpt-chemical` — BERT (apenas codificador)
treinado para reconhecimento de entidades nomeadas em textos clínicos em
português. Possui 110 milhões de parâmetros.

### Por que apenas codificador (encoder-only)?

- Atenção **bidirecional**: cada token "enxerga" tokens à esquerda E à direita
- Ideal para tarefas de **compreensão**: classificação, reconhecimento de
  entidades, geração de embeddings
- Arquitetura base: 12 camadas Transformers, 768 dimensões de estado oculto
- Tokenização WordPiece: palavras frequentes viram tokens únicos; palavras
  raras são quebradas em sub-tokens (exemplo: "poderoso" → "poder", "##oso")
```

### Célula 4 — AutoModel na prática (Code)

```python
identificador_modelo = modelo_reconhecimento_entidades
registro.info("Carregando tokenizador e modelo: %s", identificador_modelo)

# ─── Tokenizador ─────────────────────────────────────────────────
tokenizador = AutoTokenizer.from_pretrained(identificador_modelo)
registro.info("Tokenizador carregado: vocabulário com %d tokens",
              tokenizador.vocab_size)

# ─── Modelo (apenas codificador, sem cabeçalho de tarefa) ────────
modelo_codificador = AutoModel.from_pretrained(identificador_modelo).to(dispositivo)
registro.info("Modelo carregado e movido para: %s", dispositivo)

# ─── Processamento da entrada (estilo do professor) ──────────────
frase_exemplo = "O mecanismo de atenção é poderoso"
entradas = tokenizador(frase_exemplo, return_tensors="pt")
entradas = {chave: valor.to(dispositivo) for chave, valor in entradas.items()}

# ─── Propagação direta ───────────────────────────────────────────
with torch.no_grad():
    saidas = modelo_codificador(**entradas)

# ─── Inspeção das dimensões ──────────────────────────────────────
dimensoes_estado_oculto = saidas.last_hidden_state.shape
registro.info(
    "Dimensões do estado oculto: [Lote=%d, Tokens=%d, Dimensão=%d]",
    dimensoes_estado_oculto[0],
    dimensoes_estado_oculto[1],
    dimensoes_estado_oculto[2]
)

print(f"Dimensões do estado oculto: {dimensoes_estado_oculto}")
print(f"  Interpretação:")
print(f"  • Lote (Batch):     {dimensoes_estado_oculto[0]} (uma frase)")
print(f"  • Tokens:           {dimensoes_estado_oculto[1]} (incluindo [CLS] e [SEP])")
print(f"  • Dimensão oculta:  {dimensoes_estado_oculto[2]} (embedding BERT)")

# ─── Exibir tokens WordPiece ─────────────────────────────────────
tokens_gerados = tokenizador.convert_ids_to_tokens(
    entradas["input_ids"][0]
)
print(f"\nTokens WordPiece: {tokens_gerados}")
print(f"Total de tokens: {len(tokens_gerados)} (limite do BERT: 512)")

registro.info("Tokens gerados: %s", tokens_gerados)
registro.info("Total de tokens: %d", len(tokens_gerados))
```

### Célula 5 — Explicação análise de sentimento (Markdown)

```markdown
## 2.2 Análise de Sentimento em Frases Clínicas

Usamos o pipeline padrão de análise de sentimento do HuggingFace para classificar
frases reais extraídas de bulas médicas. O objetivo **não é** obter resultados
perfeitos — é **demonstrar as limitações** de um modelo genérico treinado em
críticas de filmes quando aplicado a domínio especializado.

O modelo padrão (DistilBERT ajustado no SST-2) classifica o **tom emocional**
do texto (POSITIVO/NEGATIVO), não o significado clínico. Isso ficará evidente
nos exemplos abaixo e **motiva** o uso de modelos especializados como o
`clinicalnerpt-chemical` e, futuramente, o ajuste fino do BioBERTpt.

### Por que isso importa para o projeto?

Se usássemos análise de sentimento genérica para classificar interações
medicamentosas, frases como "aumenta a toxicidade" seriam classificadas como
NEGATIVO — o que está correto por acaso. Mas "recomenda-se monitoramento"
também poderia ser NEGATIVO, quando na verdade indica uma interação LEVE que
requer apenas acompanhamento. A nuance se perde completamente.
```

### Célula 6 — Análise de sentimento na prática (Code)

```python
registro.info("Carregando pipeline de análise de sentimento...")
classificador_sentimento = pipeline("sentiment-analysis")

# Frases reais extraídas de bulas médicas brasileiras
frases_clinicas = [
    "O uso concomitante é contraindicado devido ao risco de arritmia fatal.",
    "Não há interações conhecidas com este medicamento.",
    "Recomenda-se monitoramento da função renal durante o tratamento.",
    "A administração concomitante de Amoxicilina com Metotrexato pode aumentar a toxicidade.",
    "O medicamento é seguro e bem tolerado pela maioria dos pacientes.",
]

print(f"{'Sentimento':>12} | {'Confiança':>9} | Frase")
print("-" * 75)

for frase in frases_clinicas:
    resultado = classificador_sentimento(frase)[0]
    sentimento = resultado["label"]
    confianca = resultado["score"]
    print(f"{sentimento:>12} | {confianca:>8.3f} | {frase[:55]}...")
    registro.info(
        "Sentimento: %s (confiança=%.3f) | Frase: %s",
        sentimento, confianca, frase[:80]
    )

registro.info("Análise de sentimento concluída: %d frases classificadas",
              len(frases_clinicas))
```

### Célula 7 — Explicação NER (Markdown)

```markdown
## 2.3 Reconhecimento de Entidades Nomeadas com clinicalnerpt-chemical

**Reconhecimento de Entidades Nomeadas (NER)** é uma tarefa de **classificação
de tokens**: cada token recebe um rótulo indicando se pertence a uma entidade.

O modelo `pucpr/clinicalnerpt-chemical` é um BERT ajustado especificamente para
identificar nomes de medicamentos em textos clínicos em português. Ele reconhece
tanto **princípios ativos** (Amoxicilina, Alopurinol) quanto **nomes comerciais**
(AAS Protect, Zarator, Zocor).

Usamos `aggregation_strategy="simple"` para agrupar sub-tokens. Exemplo:
`Amoxi` + `##cilina` → `Amoxicilina` (entidade única).

### Por que NER e não expressões regulares?

Nomes de medicamentos têm altíssima variabilidade: marcas, genéricos, compostos,
sufixos de sal (cloridrato, sódico, potássico). Uma expressão regular que cubra
todos os casos seria inviável de manter. O modelo aprende padrões contextuais:
"a administração concomitante de [MEDICAMENTO] com [MEDICAMENTO]".

### Rótulos do modelo

- `B-ChemicalDrugs` — início de um nome de medicamento (Beginning)
- `I-ChemicalDrugs` — continuação de um nome de medicamento (Inside)
- `O` — token fora de qualquer entidade (Outside)
```

### Célula 8 — NER na prática (Code)

```python
registro.info("Carregando pipeline de reconhecimento de entidades...")

reconhecedor_entidades = pipeline(
    "ner",
    model=modelo_reconhecimento_entidades,
    aggregation_strategy="simple",
    device=0 if dispositivo == "cuda" else -1,
)
registro.info("Modelo NER carregado: %s", modelo_reconhecimento_entidades)

# Trecho real de bula da ANVISA (amoxicilina profissional)
trecho_bula_amoxicilina = (
    "A probenecida reduz a secrecao tubular renal da amoxicilina. "
    "No uso concomitante com amoxicilina, pode haver aumento dos niveis "
    "de amoxicilina no sangue. A administracao concomitante de alopurinol "
    "durante o tratamento com amoxicilina pode aumentar a probabilidade "
    "de reacoes alergicas da pele. Existem casos raros de INR aumentada "
    "em pacientes mantidos com acenocumarol ou varfarina."
)

registro.info("Executando NER no trecho de bula (%d caracteres)...",
              len(trecho_bula_amoxicilina))

entidades_reconhecidas = reconhecedor_entidades(trecho_bula_amoxicilina)

# ─── Exibição formatada ──────────────────────────────────────────
print(f"{'Entidade':<22} {'Confiança':>9}  {'Início':>6}  {'Fim':>6}")
print("-" * 52)

for entidade in entidades_reconhecidas:
    nome_entidade = entidade["word"]
    pontuacao_confianca = entidade["score"]
    posicao_inicio = entidade["start"]
    posicao_fim = entidade["end"]
    print(f"{nome_entidade:<22} {pontuacao_confianca:>8.3f}  "
          f"{posicao_inicio:>6}  {posicao_fim:>6}")

# ─── Medicamentos únicos identificados ───────────────────────────
medicamentos_unicos = sorted(set(
    entidade["word"] for entidade in entidades_reconhecidas
))

print(f"\nMedicamentos únicos identificados ({len(medicamentos_unicos)}):")
for medicamento in medicamentos_unicos:
    print(f"  • {medicamento}")

registro.info(
    "NER concluído: %d ocorrências, %d medicamentos únicos: %s",
    len(entidades_reconhecidas),
    len(medicamentos_unicos),
    medicamentos_unicos
)
```

### Célula 9 — Explicação da tabela comparativa (Markdown)

```markdown
## 2.4 Tabela Comparativa de Arquiteturas de Modelos

### Modelos discutidos neste caderno

| Modelo | Arquitetura | Parâmetros | Tarefa Principal | Limite de Tokens | Domínio |
|---|---|---|---|---|---|
| `clinicalnerpt-chemical` | BERT (apenas codificador) | 110M | NER | 512 | Clínico PT |
| DistilBERT (sentiment) | BERT (apenas codificador) | 66M | Sentimento | 512 | Genérico EN |
| `biobertpt-all` † | BERT (apenas codificador) | 110M | Classificação | 512 | Biomédico PT |
| `gpt2-small-portuguese` ‡ | GPT-2 (apenas decodificador) | 124M | Geração | 1024 | Genérico PT |
| `bart-large-cnn` ‡ | BART (codificador-decodificador) | 406M | Sumarização | 1024 | Genérico EN |

> † Citado como referência para ajuste fino futuro.  
> ‡ Citado para contraste de arquitetura.

### Diferenças fundamentais entre as três arquiteturas

#### Apenas codificador (Encoder-only) — BERT

- **Atenção bidirecional:** cada token enxerga TODOS os outros tokens da frase
- **Treinamento:** Modelagem de Linguagem Mascarada (prever tokens ocultos)
- **Exemplo:** "O [MÁSCARA] é contraindicado" → modelo usa contexto dos dois lados
- **Ideal para:** classificação, NER, embeddings, perguntas e respostas extrativas

#### Apenas decodificador (Decoder-only) — GPT-2

- **Atenção unidirecional/causal:** cada token só enxerga tokens ANTERIORES
- **Treinamento:** Previsão do Próximo Token (Next Token Prediction)
- **Exemplo:** "O uso concomitante" → modelo prevê "é" → depois "contraindicado"...
- **Ideal para:** geração de texto, chatbots, completamento

#### Codificador-decodificador (Encoder-decoder) — BART

- **Codificador:** processa entrada bidirecionalmente (como BERT)
- **Decodificador:** gera saída autoregressivamente (como GPT-2)
- **Ideal para:** tradução, sumarização (tarefas de transformação)

### Pipeline vs Inferência Manual

| Abordagem | Vantagens | Desvantagens |
|---|---|---|
| `pipeline("ner", model=...)` | Uma linha, tokenização + modelo + pós-processamento automáticos | Menos controle, difícil depurar |
| `AutoModel` + `tokenizador` manual | Controle total, inferência em lote, GPU explícita | Aproximadamente 5 linhas por tarefa |

**Recomendação:** Use `pipeline()` para prototipagem rápida (cadernos 01-04).
Use `AutoModel` para o pipeline RAG (caderno 05), onde precisamos de controle
fino sobre GPU, processamento em lote e tratamento de erros.
```

### Célula 10 — Conclusão (Markdown)

```markdown
## 2.5 Conclusão: Quais Tarefas Importam para o Detector de Interações?

| Tarefa | Aplicação no Projeto | Onde aparece |
|---|---|---|
| **Reconhecimento de Entidades Nomeadas** | Extrair medicamentos da consulta do usuário e dos trechos das bulas | Caderno 05 (RAG) |
| **Classificação de Texto** | Classificar interação como 0 (SEM), 1 (LEVE) ou 2 (GRAVE) | Cadernos 02, 05 |
| **Geração de Embeddings** | Representar trechos de bulas para busca vetorial no FAISS | Cadernos 03, 05 |
| **Geração de Texto (LLM)** | Produzir resposta final fundamentada nos trechos recuperados | Cadernos 02, 05 |

### O que aprendemos neste caderno

1. **AutoModel + AutoTokenizer:** Como carregar e inspecionar modelos pré-treinados,
   seguindo o estilo do professor
2. **Limitação de modelos genéricos:** Análise de sentimento captura tom emocional,
   não significado clínico — justifica o uso de modelos especializados
3. **NER com clinicalnerpt-chemical:** Extrai medicamentos de bulas reais com
   alta precisão, incluindo princípios ativos e nomes comerciais
4. **Arquiteturas diferentes para tarefas diferentes:** BERT para compreensão,
   GPT-2 para geração, BART para transformação

### Próximos passos

- **Caderno 02:** Engenharia de prompt — testar zero-shot, few-shot e
  cadeia de pensamento para classificação de interações com GPT4All
- **Caderno 03:** Embeddings e busca vetorial — indexar bulas no FAISS
- **Caderno 05:** Pipeline RAG completo — integrar NER + busca + GPT4All
```

### Célula 11 — Finalização (Code)

```python
# ─── Resumo final ─────────────────────────────────────────────────
registro.info("=" * 70)
registro.info("Caderno 01 concluído com sucesso")
registro.info("Término da execução: %s", datetime.now().isoformat())
registro.info("Log completo disponível em: %s", arquivo_log.resolve())

print(f"\n✅ Caderno 01 concluído.")
print(f"📄 Log completo: {arquivo_log.resolve()}")
```

---

## 3. Execução e Verificação

### 3.1 Como executar

```bash
cd C:\workspace\python\projeto-2-modulo-1-pos
source venv/Scripts/activate
jupyter notebook c01_modelos_llm.ipynb
```

Dentro do Jupyter: Kernel → Restart & Run All.

### 3.2 O que verificar

| Verificação | Como verificar | Esperado |
|---|---|---|
| GPU funcionando | Célula 2, output | `CUDA disponível: True` |
| AutoModel carregou | Célula 4, output | `Dimensões do estado oculto: torch.Size([1, 8, 768])` |
| NER extraiu entidades | Célula 8, output | Pelo menos 3 medicamentos únicos listados |
| Logs gerados | `cat logs/caderno_01_modelos_linguagem.log` | Linhas com timestamps, INFO, nomes de modelos |
| Nenhum erro | Jupyter, barra de status | Sem células com erro (sem vermelho) |

### 3.3 Exemplo de log esperado

```
2026-06-21 14:00:01 [INFO] ======================================================================
2026-06-21 14:00:01 [INFO] Caderno 01 — Modelos de Linguagem e Processamento de Linguagem Natural
2026-06-21 14:00:01 [INFO] Início da execução: 2026-06-21T14:00:01
2026-06-21 14:00:01 [INFO] PyTorch versão: 2.6.0+cu124
2026-06-21 14:00:01 [INFO] CUDA disponível: True
2026-06-21 14:00:01 [INFO] Dispositivo configurado: cuda
2026-06-21 14:00:01 [INFO] GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU | VRAM total: 6.4 GB
2026-06-21 14:00:05 [INFO] Carregando tokenizador e modelo: pucpr/clinicalnerpt-chemical
2026-06-21 14:00:05 [INFO] Tokenizador carregado: vocabulário com 29794 tokens
2026-06-21 14:00:05 [INFO] Modelo carregado e movido para: cuda
2026-06-21 14:00:05 [INFO] Dimensões do estado oculto: [Lote=1, Tokens=8, Dimensão=768]
2026-06-21 14:00:05 [INFO] Tokens gerados: ['[CLS]', 'O', 'mecanismo', 'de', 'atencao', 'e', 'poderoso', '[SEP]']
2026-06-21 14:00:05 [INFO] Total de tokens: 8
2026-06-21 14:00:08 [INFO] Carregando pipeline de análise de sentimento...
2026-06-21 14:00:08 [INFO] Análise de sentimento concluída: 5 frases classificadas
2026-06-21 14:00:12 [INFO] Carregando pipeline de reconhecimento de entidades...
2026-06-21 14:00:12 [INFO] Modelo NER carregado: pucpr/clinicalnerpt-chemical
2026-06-21 14:00:12 [INFO] NER concluído: 7 ocorrências, 4 medicamentos únicos: ['acenocumarol', 'alopurinol', 'amoxicilina', 'varfarina']
2026-06-21 14:00:12 [INFO] ======================================================================
2026-06-21 14:00:12 [INFO] Caderno 01 concluído com sucesso
```

---

## 4. Commits Atômicos

### Commit 1: Estrutura e requisitos
```
chore: Sprint 1 — estrutura do projeto, requirements.txt, .gitignore
```
Arquivos: `requirements.txt`, `.gitignore`

### Commit 2: Células 1-6 (setup + AutoModel + sentimento)
```
feat: Sprint 1 — caderno 01 células 1-6: setup, logging, AutoModel, análise de sentimento
```
Arquivos: `c01_modelos_llm.ipynb` (parcial)

### Commit 3: Células 7-8 (NER)
```
feat: Sprint 1 — caderno 01 células 7-8: reconhecimento de entidades nomeadas com clinicalnerpt-chemical
```
Arquivos: `c01_modelos_llm.ipynb` (atualizado)

### Commit 4: Células 9-11 (tabela + conclusão + finalização)
```
feat: Sprint 1 — caderno 01 células 9-11: tabela comparativa de arquiteturas, conclusão, finalização
```
Arquivos: `c01_modelos_llm.ipynb` (final)
