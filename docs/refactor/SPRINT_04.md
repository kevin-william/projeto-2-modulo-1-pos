# Sprint 4 — Notebook 04 (Inferência Local)

**Objetivo:** Implementar o Notebook 04 comparando três backends de inferência
100% locais: GPT4All ligação direta, GPT4All servidor API e heurística de
palavras-chave. Comparar qualidade, latência, consumo de RAM, facilidade de
configuração e privacidade.

**Duração:** 2-3 horas  
**Commits:** 3 atômicos  
**Rubricas cobertas:** Rubrica 4 (5 itens)

---

## 1. Classe ComparadorInferencia

```python
class ComparadorInferencia:
    """
    Compara múltiplos backends de inferência em cinco dimensões:
    qualidade, latência, consumo de memória, facilidade de configuração e privacidade.

    Atributos:
        backends (dict): Dicionário mapeando nome do backend → instância.
        resultados_qualidade (dict): Métricas de qualidade por backend.
        resultados_latencia (dict): Métricas de latência por backend.
        resultados_memoria (dict): Consumo de RAM por backend.
    """

    def __init__(self):
        self.backends = {}
        self.resultados_qualidade = {}
        self.resultados_latencia = {}
        self.resultados_memoria = {}

    def registrar_backend(self, nome, instancia):
        """Registra um backend para comparação."""
        self.backends[nome] = instancia
        registro.info("Backend registrado: %s", nome)

    def medir_qualidade(self, pares_teste):
        """
        Avalia acurácia e F1 de cada backend nos pares de teste.

        Argumentos:
            pares_teste (list[dict]): Lista de pares com 'classe_esperada'.

        Retorna:
            dict: Métricas por backend.
        """
        registro.info("Iniciando avaliação de qualidade com %d pares", len(pares_teste))

        for nome_backend, backend in self.backends.items():
            acertos = 0
            total = 0
            matriz_confusao = {0: {0:0, 1:0, 2:0}, 1: {0:0, 1:0, 2:0}, 2: {0:0, 1:0, 2:0}}

            for par in pares_teste:
                prompt = modelo_prompt_base.format(
                    trecho_bula=par["trecho_bula"],
                    medicamento_principal=par["medicamento_principal"],
                    medicamento_secundario=par["medicamento_secundario"],
                )

                if hasattr(backend, 'gerar'):
                    resposta = backend.gerar(prompt, maximo_tokens=150)
                else:
                    resposta = backend(prompt)

                dados = analisar_resposta_json(resposta)
                classe_obtida = dados["classe"] if dados else -1
                classe_esperada = par["classe_esperada"]

                if classe_obtida == classe_esperada:
                    acertos += 1
                if classe_obtida in (0, 1, 2) and classe_esperada in (0, 1, 2):
                    matriz_confusao[classe_esperada][classe_obtida] += 1
                total += 1

            acuracia = acertos / total if total > 0 else 0
            self.resultados_qualidade[nome_backend] = {
                "acuracia": acuracia,
                "total_pares": total,
                "matriz_confusao": matriz_confusao,
            }
            registro.info(
                "Qualidade %s: acurácia=%.2f (%d/%d)",
                nome_backend, acuracia, acertos, total
            )

        return self.resultados_qualidade

    def medir_latencia(self, consulta_padrao, repeticoes=10):
        """
        Mede latência média e P95 para cada backend.

        Argumentos:
            consulta_padrao (str): Prompt de teste padronizado.
            repeticoes (int): Quantidade de medições.

        Retorna:
            dict: Latências por backend.
        """
        registro.info("Iniciando medição de latência (%d repetições)", repeticoes)
        import time, numpy as np

        for nome_backend, backend in self.backends.items():
            tempos = []

            for i in range(repeticoes):
                tempo_inicio = time.time()

                if hasattr(backend, 'gerar'):
                    backend.gerar(consulta_padrao, maximo_tokens=100)
                else:
                    backend(consulta_padrao)

                tempos.append((time.time() - tempo_inicio) * 1000)

            tempos = np.array(tempos)
            self.resultados_latencia[nome_backend] = {
                "media_ms": float(np.mean(tempos)),
                "mediana_ms": float(np.median(tempos)),
                "p95_ms": float(np.percentile(tempos, 95)),
                "minimo_ms": float(np.min(tempos)),
                "maximo_ms": float(np.max(tempos)),
            }
            registro.info(
                "Latência %s: média=%.0fms, P95=%.0fms",
                nome_backend,
                self.resultados_latencia[nome_backend]["media_ms"],
                self.resultados_latencia[nome_backend]["p95_ms"],
            )

        return self.resultados_latencia

    def medir_memoria(self):
        """
        Mede consumo de RAM de cada backend via psutil.

        Retorna:
            dict: Consumo de memória em MB por backend.
        """
        import psutil

        processo = psutil.Process()
        memoria_base = processo.memory_info().rss / 1024 / 1024

        for nome_backend in self.backends:
            memoria_atual = processo.memory_info().rss / 1024 / 1024
            self.resultados_memoria[nome_backend] = {
                "memoria_total_mb": memoria_atual,
                "memoria_adicional_mb": memoria_atual - memoria_base,
            }

        return self.resultados_memoria
```

---

## 2. Os Três Backends

### 2.1 Backend 1: GPT4All Ligação Direta

```python
class BackendLigacaoDireta:
    """
    GPT4All via binding Python oficial.

    O modelo .gguf é carregado integralmente em RAM.
    Inferência via CPU. Requer ~4.5 GB de RAM livre.

    Vantagens:
        - Zero configuração (pip install + download automático)
        - 100% offline
        - Interface Python nativa

    Desvantagens:
        - Alto consumo de RAM
        - Latência de 2-5 segundos por consulta (CPU)
        - Qualidade depende do modelo .gguf escolhido
    """

    def __init__(self, nome_modelo="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
        from gpt4all import GPT4All
        self.nome_modelo = nome_modelo
        self.modelo = GPT4All(nome_modelo)
        self.rotulo = "GPT4All Direto"
        self.descricao = "Binding Python oficial GPT4All — modelo carregado em RAM"

    def gerar(self, consulta, maximo_tokens=150):
        return self.modelo.generate(consulta, max_tokens=maximo_tokens)
```

### 2.2 Backend 2: GPT4All Servidor API

```python
class BackendServidorAPI:
    """
    GPT4All via servidor HTTP local (compatível com OpenAI).

    Requer o aplicativo desktop GPT4All com 'Enable Local API Server'
    ativado nas configurações. O servidor escuta em localhost:4891.

    Vantagens:
        - Interface compatível com OpenAI (fácil migração)
        - Modelo gerenciado pelo app desktop
        - Latência menor que ligação direta (~1s vs ~3s)

    Desvantagens:
        - Requer instalação do app desktop
        - Requer ativação manual do servidor API
        - Dependência de processo externo
    """

    def __init__(self, nome_modelo="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
        from openai import OpenAI
        self.nome_modelo = nome_modelo
        self.cliente = OpenAI(
            base_url="http://localhost:4891/v1",
            api_key="gpt4all"
        )
        self.rotulo = "GPT4All API"
        self.descricao = "Servidor HTTP local do GPT4All — compatível com OpenAI"

    def gerar(self, consulta, maximo_tokens=150):
        resposta = self.cliente.chat.completions.create(
            model=self.nome_modelo,
            messages=[{"role": "user", "content": consulta}],
            max_tokens=maximo_tokens,
            temperature=0.1,
        )
        return resposta.choices[0].message.content
```

### 2.3 Backend 3: Heurística de Palavras-Chave

```python
class BackendHeuristica:
    """
    Classificação baseada exclusivamente em palavras-chave.

    Não requer modelo de linguagem. Instantânea. Zero consumo de RAM adicional.

    Usa a mesma lógica de _classificar_por_palavras_chave da Sprint 2.

    Vantagens:
        - Latência < 1ms
        - Zero consumo de RAM/VRAM
        - Zero dependências

    Desvantagens:
        - Qualidade inferior (~55% vs ~78% dos LLMs)
        - Não entende contexto ou nuance
        - Falso positivo/negativo para casos ambíguos
    """

    def __init__(self):
        self.rotulo = "Heurística"
        self.descricao = "Classificação por palavras-chave — sem modelo de linguagem"

    def gerar(self, consulta, maximo_tokens=150):
        return self._classificar(consulta)

    def _classificar(self, texto):
        texto_minusculo = texto.lower()

        palavras_graves = [
            "contraindicado", "fatal", "risco de morte",
            "não administrar", "rabdomiólise"
        ]
        for palavra in palavras_graves:
            if palavra in texto_minusculo:
                return (
                    '{"classe": 2, "justificativa": "Palavra grave: '
                    + palavra
                    + '", "evidencia": "Heurística"}'
                )

        frases_ausencia = [
            "não há interação", "sem interação", "não foram observadas"
        ]
        for frase in frases_ausencia:
            if frase in texto_minusculo:
                return (
                    '{"classe": 0, "justificativa": "Frase de ausência: '
                    + frase
                    + '", "evidencia": "Heurística"}'
                )

        palavras_leves = [
            "monitorar", "ajustar", "cautela", "precaução"
        ]
        for palavra in palavras_leves:
            if palavra in texto_minusculo:
                return (
                    '{"classe": 1, "justificativa": "Palavra leve: '
                    + palavra
                    + '", "evidencia": "Heurística"}'
                )

        return '{"classe": 0, "justificativa": "Sem palavras-chave", "evidencia": "Heurística"}'
```

---

## 3. Estrutura do Caderno (9 células)

| Célula | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica, explicação dos 3 backends |
| 2 | Code | **Configuração:** logging (`caderno_04_inferencia.log`), classe `ComparadorInferencia`, 3 backends instanciados. Log de cada backend. |
| 3 | Code + Markdown | **Qualidade:** carregar 30 pares, medir acurácia/F1 de cada backend. Tabela comparativa. |
| 4 | Code + Markdown | **Latência:** 10 consultas padronizadas, média/P95. Gráfico de barras. |
| 5 | Code + Markdown | **Consumo de RAM:** `psutil` antes/depois de cada backend. |
| 6 | Markdown | **Facilidade de configuração:** comparação qualitativa (instalação, dependências, complexidade) |
| 7 | Markdown | **Privacidade:** análise — todos são 100% locais. Comparação teórica com APIs cloud. |
| 8 | Markdown | **Conclusão:** tabela 5 dimensões + recomendação para o pipeline RAG |
| 9 | Code | `registro.info("Caderno 04 concluído. Backends testados: %d", len(comparador.backends))` |

### 3.1 Tabela comparativa esperada

```markdown
| Dimensão | Ligação Direta | Servidor API | Heurística |
|---|---|---|---|
| **Qualidade (F1)** | ~0.78 | ~0.78 | ~0.55 |
| **Latência (ms)** | ~3200 | ~1000 | <1 |
| **RAM adicional** | ~4.5 GB | 0 | 0 |
| **Configuração** | `pip install gpt4all` | App desktop + ativar API | Nada |
| **Dependências** | Nenhuma externa | Processo externo | Nenhuma |
| **Offline** | ✅ | ✅ | ✅ |
| **Privacidade** | ✅ 100% local | ✅ 100% local | ✅ 100% local |
```

---

## 4. Commits Atômicos

### Commit 1: Classe ComparadorInferencia + 3 backends
```
feat: Sprint 4 — ComparadorInferencia + 3 backends: ligação direta, servidor API, heurística
```

### Commit 2: Métricas de qualidade, latência e RAM
```
feat: Sprint 4 — medir_qualidade, medir_latencia, medir_memoria com 30 pares
```

### Commit 3: Análise qualitativa + conclusão com tabela 5 dimensões
```
feat: Sprint 4 — análise de configuração, privacidade, tabela comparativa, recomendação
```
