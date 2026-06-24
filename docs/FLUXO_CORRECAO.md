# Fluxo de Correção de Notebooks

## Princípio

Notebooks são para exploração e demonstração — não para debug. Quando algo não funciona, o código vai para um `.py` estruturado em classe, é corrigido e testado no terminal, e só então volta ao notebook.

---

## Estrutura de Arquivos

Cada notebook tem seu próprio script de debug:

```
scripts/
  debug_c01.py   ← DebugC01
  debug_c02.py   ← DebugC02
  debug_c03.py   ← DebugC03
  debug_c04.py   ← DebugC04
  debug_c05.py   ← DebugC05
```

Cada script é uma **classe** onde cada método corresponde a uma célula do notebook. Variáveis compartilhadas no `__init__` simulam o estado do kernel.

---

## Template — como escrever um `debug_cXX.py`

```python
import sys, os
sys.path.insert(0, os.path.abspath('.'))  # permite importar scripts/

from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"]       = os.getenv("HF_TOKEN", "")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

import numpy as np
import pandas as pd
import time
import json

# Imports do projeto
from scripts.config import *
from scripts.embeddings import *
from scripts.ner import *
from scripts.classifier import *
from scripts.rag import *


class DebugC03:
    """Debug do c03_embeddings_busca.ipynb"""

    def __init__(self):
        # Variáveis que existem no kernel do notebook
        self.collection = None
        self.df_chunks  = None
        self.MODELS = {
            "MINILM": "sentence-transformers/all-MiniLM-L6-v2",
            "BERTPT": "pucpr/biobertpt-all",
        }
        # setup — roda uma vez
        self._setup()

    def _setup(self):
        """Setup comum a todas as células"""
        # carrega collection, dados, etc.
        pass

    def cell_05(self):
        """Pré-processamento — gera chunks_bulas.jsonl se não existir"""
        # ... código da célula ...
        pass

    def cell_09(self):
        """Gera embeddings com MINILM"""
        # ... código da célula ...
        pass

    def cell_13(self):
        """Avaliação — loop sobre modelos"""
        # ... código da célula ...
        pass

    def run_all(self):
        """Executa todas as células em ordem (simula 'Run All')"""
        self.cell_05()
        self.cell_09()
        self.cell_13()

    def run_cell(self, n):
        """Executa uma célula específica pelo número"""
        getattr(self, f"cell_{n:02d}")()


if __name__ == "__main__":
    import sys
    dbg = DebugC03()

    if len(sys.argv) > 1:
        dbg.run_cell(int(sys.argv[1]))
    else:
        dbg.run_all()
```

---

## Fluxo Completo

```
Notebook (c03) — célula 13 com bug (InvalidArgumentError: 384 vs 768d)
    │
    ▼
1. EXTRAIR
   Criar / atualizar scripts/debug_c03.py
   → Copiar código das células 05, 09, 11, 13 como métodos
   → No __init__, recriar o estado do kernel (collection carrega, etc.)
    │
    ▼
2. CORRIGIR
   Editar o método cell_13 no .py — trocar MODELS.keys() por ['MINILM']
    │
    ▼
3. EXECUTAR NO TERMINAL
   cd /c/workspace/python/projeto-2-modulo-1-pos
   source venv/Scripts/activate
   python scripts/debug_c03.py 13     ← só a célula 13
   python scripts/debug_c03.py        ← todas (run_all)
    │
    ▼
4. AVALIAR
   │─ Saída OK?        → ir para 5
   │─ Erro?            → voltar para 2 (corrigir)
   │─ Erro de import?  → verificar venv, kernel, dependências
   │
    ▼
5. APLICAR NO NOTEBOOK
   Copiar código corrigido do método cell_13 de volta para a célula no .ipynb
   (usar patch json — não reescrever o notebook inteiro)
    │
    ▼
6. VALIDAR
   Usuário abre o notebook no VS Code
   │─ Restart Kernel
   │─ Executa célula corrigida
   │─ OK?   → done
   │─ Erro? → nova iteração (volta para 1)
```

---

## Regra de Ouro

**Se o debug script precisa de uma variável que deveriavir de uma célula anterior (ex: `self.collection`), recria-la no `__init__`** — não pular células. O objetivo é que `run_all()` reproduza o estado real do kernel executando todas as células em ordem.

---

## Regras

- **Scripts de debug vivem em `scripts/debug_cXX.py`** — não commitar ao final
- **Manter o venv ativado** — `source venv/Scripts/activate` antes de rodar
- **Testar com dados reais do projeto** — não mockar mais que o necessário
- **O notebook nunca é executado pelo agente** — só editado
- **Todas as células dependem de `from scripts.config import *`** — garantir que o kernel do VS Code aponta para o venv
- **Antes de começar o debug**: ler as fases do FASE_N.md correspondentes para entender o contexto da tarefa

---

## Comandos Úteis

```bash
# Executar todas as células (simula Run All)
python scripts/debug_c03.py

# Executar só célula 13
python scripts/debug_c03.py 13

# Executar de 09 até 13
python scripts/debug_c03.py 09 13
```

---

## Exemplo — Debug C03 Célula 13

```python
# Em scripts/debug_c03.py

class DebugC03:
    # ... __init__ e outros métodos ...

    def cell_13(self):
        """Comparação de modelos (BUG: iterava todos os modelos mas
        collection só tem embeddings MINILM 384d)"""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        CONSULTAS_TESTE = [
            ('sinvastatina itraconazol contraindicado', ['cvar_087', 'sin_099']),
            ('amoxicilina alergia penicilina', ['amox_012', 'pen_003']),
            # ... 10 consultas ...
        ]

        def metricas_busca(collection, modelo_key, consultas, n=3):
            model = SentenceTransformer(
                self.MODELS[modelo_key],
                cache_folder='data/modelos_cache'
            )
            p_scores, mrr_scores, latencias = [], [], []

            for consulta, ids_esperados in consultas:
                t0 = time.time()
                emb = model.encode([consulta], normalize_embeddings=True)[0]
                top_n = buscar_chunks(collection, emb, n=n)
                lat = (time.time() - t0) * 1000

                ids_ret = [r['id'] for r in top_n]
                p = len(set(ids_ret) & set(ids_esperados)) / n
                mrr = 0.0
                for rank, rid in enumerate(ids_ret, 1):
                    if rid in ids_esperados:
                        mrr = 1.0 / rank
                        break

                p_scores.append(p)
                mrr_scores.append(mrr)
                latencias.append(lat)

            return {
                'modelo': modelo_key,
                'dims': model.get_sentence_embedding_dimension(),
                'p_at_3': round(np.mean(p_scores), 3),
                'mrr': round(np.mean(mrr_scores), 3),
                'lat_ms': round(np.mean(latencias), 1),
            }

        print('Comparando modelos...')
        resultados = []

        # BUG CORRIGIDO: collection foi indexada só com MINILM (384d)
        # iterar BERTPT (768d) causava InvalidArgumentError
        for nome in ['MINILM']:
            r = metricas_busca(self.collection, nome, CONSULTAS_TESTE)
            resultados.append(r)
            print(f"  {r['modelo']}: P@3={r['p_at_3']} | MRR={r['mrr']} | Lat={r['lat_ms']}ms | {r['dims']}d")

        print("Pronto!")
```
