# Sprint 5 — Notebook 05 (Pipeline RAG Completo)

**Objetivo:** Implementar o Notebook 05 integrando NER, busca vetorial (FAISS)
e GPT4All em um pipeline RAG que recebe consulta em linguagem natural e retorna
JSON estruturado com classificação, justificativa, evidência e fonte.

**Duração:** 4-5 horas  
**Commits:** 6 atômicos  
**Rubricas cobertas:** Rubrica 5 (9 dos 11 itens — os 2 restantes na Sprint 6)

---

## 1. Classe PipelineInteracao (Coração do Sistema)

```python
class PipelineInteracao:
    """
    Pipeline completo de detecção de interações medicamentosas.

    Fluxo de execução para cada consulta:
    1. NER → extrai medicamentos da consulta do usuário
    2. Para cada par de medicamentos:
       a. Busca trechos relevantes no FAISS (cosseno)
       b. Filtra trechos que mencionam AMBOS os medicamentos
       c. Constrói prompt Few-shot com os trechos recuperados
       d. Classifica via GPT4All (com fallback)
       e. Analisa JSON da resposta
    3. Consolida todos os pares em JSON estruturado

    Atributos:
        reconhecedor_entidades: Pipeline NER do HuggingFace.
        modelo_embeddings: Modelo SentenceTransformer para embeddings.
        indice_faiss: Índice FAISS com os trechos das bulas.
        trechos: Lista de trechos indexados.
        provedor_linguagem: Instância de ProvedorLinguagem (GPT4All).
        consultas_realizadas (int): Contador de consultas processadas.
    """

    def __init__(self):
        self.reconhecedor_entidades = None
        self.modelo_embeddings = None
        self.indice_faiss = None
        self.trechos = None
        self.provedor_linguagem = None
        self.consultas_realizadas = 0

    def inicializar(self, diretorio_bulas, maximo_arquivos=500):
        """
        Carrega todos os modelos e indexa as bulas.

        Etapas:
        1. Carregar NER (clinicalnerpt-chemical, GPU)
        2. Carregar embeddings (bert-base-portuguese-cased, GPU)
        3. Carregar e indexar bulas no FAISS
        4. Inicializar GPT4All

        Argumentos:
            diretorio_bulas (Path): Diretório com fonte1/ e fonte2/.
            maximo_arquivos (int): Limite de arquivos por fonte.
        """
        registro.info("=" * 70)
        registro.info("Inicializando Pipeline de Interação Medicamentosa")
        tempo_inicio = __import__("time").time()

        # Etapa 1: NER
        registro.info("Etapa 1/4: Carregando reconhecedor de entidades...")
        self.reconhecedor_entidades = pipeline(
            "ner",
            model="pucpr/clinicalnerpt-chemical",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )
        registro.info("  NER carregado: clinicalnerpt-chemical")

        # Etapa 2: Embeddings
        registro.info("Etapa 2/4: Carregando modelo de embeddings...")
        self.modelo_embeddings = SentenceTransformer(
            "neuralmind/bert-base-portuguese-cased"
        )
        if torch.cuda.is_available():
            self.modelo_embeddings = self.modelo_embeddings.to("cuda")
        registro.info("  Embeddings carregado: bert-base-portuguese-cased")

        # Etapa 3: Indexar bulas
        registro.info("Etapa 3/4: Carregando e indexando bulas...")
        self.trechos = carregar_trechos_bulas(diretorio_bulas, maximo_arquivos)
        self.indice_faiss, _ = criar_indice_busca(
            self.trechos, self.modelo_embeddings
        )
        registro.info("  FAISS: %d vetores indexados", self.indice_faiss.ntotal)

        # Etapa 4: GPT4All
        registro.info("Etapa 4/4: Inicializando provedor de linguagem...")
        self.provedor_linguagem = ProvedorLinguagem()
        registro.info("  Backend ativo: %s", self.provedor_linguagem.camada_ativa)

        tempo_total = (__import__("time").time() - tempo_inicio)
        registro.info("Pipeline inicializado em %.1f segundos", tempo_total)
        registro.info("=" * 70)

    def consultar(self, consulta_usuario):
        """
        Processa uma consulta em linguagem natural e retorna interações detectadas.

        Exemplo:
            >>> pipeline.consultar("Posso tomar Amoxicilina com Ibuprofeno?")
            {
                "consulta": "Posso tomar Amoxicilina com Ibuprofeno?",
                "medicamentos_encontrados": ["amoxicilina", "ibuprofeno"],
                "interacoes": [
                    {
                        "medicamento_principal": "amoxicilina",
                        "medicamento_secundario": "ibuprofeno",
                        "classe": 1,
                        "classe_nome": "LEVE_MODERADA",
                        ...
                    }
                ],
                "tempo_total_ms": 4200
            }

        Argumentos:
            consulta_usuario (str): Consulta em linguagem natural.

        Retorna:
            dict: Resultado estruturado com interações detectadas.
        """
        self.consultas_realizadas += 1
        tempo_inicio = __import__("time").time()

        registro.info("-" * 60)
        registro.info("Consulta #%d: '%s'", self.consultas_realizadas, consulta_usuario)

        # ── ETAPA 1: NER ──────────────────────────────────────────
        registro.info("  Etapa 1/3: Reconhecendo entidades...")
        entidades_brutas = self.reconhecedor_entidades(consulta_usuario)
        medicamentos_encontrados = sorted(set(
            entidade["word"].lower() for entidade in entidades_brutas
        ))
        registro.info("  Medicamentos encontrados: %s", medicamentos_encontrados)

        if len(medicamentos_encontrados) < 2:
            registro.warning("  Apenas %d medicamento(s) — impossível formar par",
                           len(medicamentos_encontrados))
            return {
                "consulta": consulta_usuario,
                "erro": (
                    "Especifique pelo menos dois medicamentos para verificar interação."
                    if len(medicamentos_encontrados) == 1
                    else "Não identifiquei medicamentos na sua consulta."
                ),
                "medicamentos_encontrados": medicamentos_encontrados,
            }

        # ── ETAPA 2: BUSCA + CLASSIFICAÇÃO ────────────────────────
        registro.info("  Etapa 2/3: Buscando e classificando interações...")
        interacoes_encontradas = []

        for indice_principal in range(len(medicamentos_encontrados)):
            for indice_secundario in range(indice_principal + 1,
                                           len(medicamentos_encontrados)):
                medicamento_principal = medicamentos_encontrados[indice_principal]
                medicamento_secundario = medicamentos_encontrados[indice_secundario]

                resultado = self._analisar_par(
                    medicamento_principal, medicamento_secundario
                )
                if resultado:
                    interacoes_encontradas.append(resultado)

        # ── ETAPA 3: CONSOLIDAÇÃO ─────────────────────────────────
        tempo_total = (__import__("time").time() - tempo_inicio) * 1000
        registro.info(
            "  Etapa 3/3: Concluído em %.0fms. %d interações encontradas.",
            tempo_total, len(interacoes_encontradas)
        )

        return {
            "consulta": consulta_usuario,
            "medicamentos_encontrados": medicamentos_encontrados,
            "interacoes": interacoes_encontradas,
            "quantidade_interacoes": len(interacoes_encontradas),
            "tempo_total_ms": round(tempo_total, 0),
            "backend_linguagem": self.provedor_linguagem.camada_ativa,
        }

    def _analisar_par(self, medicamento_principal, medicamento_secundario):
        """
        Analisa a interação entre um par de medicamentos.

        1. Busca trechos relevantes no FAISS
        2. Filtra trechos que mencionam ambos os medicamentos
        3. Constrói prompt Few-shot
        4. Classifica via GPT4All
        5. Analisa JSON da resposta
        """
        # Buscar trechos
        consulta_busca = (
            f"{medicamento_principal} {medicamento_secundario} interação"
        )
        embedding_consulta = self.modelo_embeddings.encode(
            [consulta_busca], normalize_embeddings=True
        ).astype(np.float32)

        distancias, indices = self.indice_faiss.search(embedding_consulta, 10)

        # Filtrar trechos que mencionam AMBOS os medicamentos
        trechos_relevantes = []
        for distancia, indice in zip(distancias[0], indices[0]):
            trecho = self.trechos[indice]
            texto_minusculo = trecho["texto"].lower()
            if (medicamento_principal in texto_minusculo
                    and medicamento_secundario in texto_minusculo):
                trechos_relevantes.append({
                    "trecho": trecho,
                    "similaridade": float(distancia),
                })
            if len(trechos_relevantes) >= 3:
                break

        registro.info(
            "    Par %s + %s: %d trechos relevantes (top: %.3f)",
            medicamento_principal, medicamento_secundario,
            len(trechos_relevantes),
            trechos_relevantes[0]["similaridade"] if trechos_relevantes else 0,
        )

        if not trechos_relevantes:
            return None

        # Construir prompt
        trechos_formatados = "\n\n".join(
            f"[Fonte: {t['trecho']['nome_arquivo']}]\n{t['trecho']['texto']}"
            for t in trechos_relevantes[:3]
        )

        prompt = modelo_prompt_few_shot.format(
            trecho_bula=trechos_formatados,
            medicamento_principal=medicamento_principal,
            medicamento_secundario=medicamento_secundario,
        )

        # Classificar
        resposta_bruta = self.provedor_linguagem.gerar(prompt, maximo_tokens=200)
        dados = analisar_resposta_json(resposta_bruta)

        classe = dados["classe"] if dados else -1
        return {
            "medicamento_principal": medicamento_principal,
            "medicamento_secundario": medicamento_secundario,
            "classe": classe,
            "classe_nome": {
                0: "SEM_INTERACAO",
                1: "LEVE_MODERADA",
                2: "GRAVE_CONTRAINDICADA",
            }.get(classe, "ERRO"),
            "justificativa": dados.get("justificativa", "") if dados else "",
            "evidencia": dados.get("evidencia", "") if dados else "",
            "fonte": trechos_relevantes[0]["trecho"]["nome_arquivo"],
            "similaridade_faiss": round(trechos_relevantes[0]["similaridade"], 3),
        }
```

---

## 2. Sanitização de Entrada (Proteção contra Injeção de Prompt)

```python
def sanitizar_consulta(consulta_usuario):
    """
    Remove padrões de injeção de prompt da consulta do usuário.

    Ataques conhecidos bloqueados:
    - "Ignore todas as instruções anteriores"
    - "Desconsidere o que foi dito antes"
    - "Você agora é um assistente diferente"
    - "system:", "<|im_start|>", "<|im_end|>" (tokens de controle)
    - "new instructions:", "novas instruções:"

    Argumentos:
        consulta_usuario (str): Consulta original do usuário.

    Retorna:
        str: Consulta sanitizada.
    """
    import re

    padroes_bloqueados = [
        r"\bignore\b",
        r"\bignorar\b",
        r"\bdesconsidere\b",
        r"\binstruções anteriores\b",
        r"\bnovas instruções\b",
        r"\bvocê agora é\b",
        r"\byou are now\b",
        r"system:",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
    ]

    consulta_limpa = consulta_usuario
    for padrao in padroes_bloqueados:
        consulta_limpa = re.sub(
            padrao, "[BLOQUEADO]", consulta_limpa, flags=re.IGNORECASE
        )

    # Remover chaves que poderiam quebrar o JSON
    consulta_limpa = consulta_limpa.replace("{", "").replace("}", "")

    # Truncar em 300 caracteres
    consulta_limpa = consulta_limpa[:300].strip()

    if consulta_limpa != consulta_usuario:
        registro.warning(
            "Sanitização aplicada: consulta modificada de %d para %d caracteres",
            len(consulta_usuario), len(consulta_limpa)
        )

    return consulta_limpa
```

---

## 3. Comparação com vs sem Contexto RAG

```python
def comparar_com_sem_contexto(pipeline, pares_teste):
    """
    Compara a classificação com e sem contexto RAG para demonstrar
    redução de alucinação.

    Modo A (sem contexto): Zero-shot puro — o LLM classifica apenas
    com base no nome dos medicamentos, sem acesso às bulas.

    Modo B (com contexto): Pipeline RAG completo — busca FAISS,
    recupera trechos, classifica com evidência.

    Esperado: Modo B tem acurácia maior e menos alucinações.
    """
    resultados = {"sem_contexto": [], "com_contexto": []}

    for par in pares_teste:
        # Modo A: sem contexto
        prompt_sem_contexto = (
            f"Classifique a interação entre {par['medicamento_principal']} "
            f"e {par['medicamento_secundario']}. Responda JSON."
        )
        resposta_sem = pipeline.provedor_linguagem.gerar(
            prompt_sem_contexto, maximo_tokens=150
        )
        dados_sem = analisar_resposta_json(resposta_sem)

        # Modo B: com contexto (pipeline completo)
        resultado_com = pipeline._analisar_par(
            par["medicamento_principal"],
            par["medicamento_secundario"],
        )

        resultados["sem_contexto"].append({
            "esperado": par["classe_esperada"],
            "obtido": dados_sem["classe"] if dados_sem else -1,
        })
        resultados["com_contexto"].append({
            "esperado": par["classe_esperada"],
            "obtido": resultado_com["classe"] if resultado_com else -1,
        })

    return resultados
```

---

## 4. Consultas de Demonstração

```python
consultas_demonstracao = [
    # 1. Interação GRAVE
    "Posso tomar Amoxicilina com Metotrexato?",

    # 2. Interação LEVE
    "Dipirona e AAS juntos têm problema?",

    # 3. SEM interação
    "Paracetamol com Amoxicilina, pode?",

    # 4. Medicamento não encontrado nas bulas
    "Invexermectina interage com Dipirona?",

    # 5. Entidade não medicamento
    "Posso beber álcool tomando Paracetamol?",

    # 6. Múltiplos medicamentos
    "Amoxicilina, Ibuprofeno e Dipirona juntos?",

    # 7. Nome comercial + princípio ativo
    "AAS Protect com Ibuprofeno é seguro?",

    # 8. Consulta ambígua (sem medicamentos claros)
    "Esses dois remédios juntos fazem mal?",
]
```

---

## 5. Estrutura do Caderno (14 células)

| Célula | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica, diagrama ASCII da arquitetura |
| 2 | Code | **Configuração:** logging (`caderno_05_pipeline.log`), imports, constantes |
| 3 | Code | `carregar_trechos_bulas()` + `criar_indice_busca()` (reutilizados das Sprints 3 e 2) |
| 4 | Code | `sanitizar_consulta()` — proteção contra injeção de prompt |
| 5 | Code | **Classe `PipelineInteracao` completa:** `inicializar()`, `consultar()`, `_analisar_par()`. `registro.info("Pipeline inicializado em %.1fs")` |
| 6 | Code | **Inicialização:** `pipeline = PipelineInteracao()`, `pipeline.inicializar(diretorio_bulas, maximo_arquivos=500)` |
| 7 | Code + Markdown | **8 consultas de demonstração.** Para cada consulta: `resultado = pipeline.consultar(consulta)` + exibição formatada do JSON. Log detalhado de cada etapa. |
| 8 | Code + Markdown | **Comparação com vs sem contexto:** executar 5 pares nos dois modos. Exemplo emblemático de alucinação reduzida. |
| 9 | Code + Markdown | **Estratégias de chunking:** comparar sentenças vs parágrafos de 3 vs parágrafos de 5. Medir Recall@5. |
| 10 | Markdown | **Análise de falhas do pipeline:** 3 cenários documentados (NER falha, trecho irrelevante, classificação errada) |
| 11 | Code + Markdown | **Demonstração de injeção de prompt:** consulta maliciosa → sanitização → resposta normal |
| 12 | Markdown | **Riscos de segurança:** prompt injection, vazamento de contexto, envenenamento de dados, alucinação |
| 13 | Markdown | **Limitações e melhorias futuras** |
| 14 | Code | `registro.info("Caderno 05 concluído. Total de consultas: %d", pipeline.consultas_realizadas)` |

---

## 6. Logs Esperados

```
2026-06-21 17:00:01 [INFO] ======================================================================
2026-06-21 17:00:01 [INFO] Inicializando Pipeline de Interação Medicamentosa
2026-06-21 17:00:01 [INFO] Etapa 1/4: Carregando reconhecedor de entidades...
2026-06-21 17:00:05 [INFO]   NER carregado: clinicalnerpt-chemical
2026-06-21 17:00:05 [INFO] Etapa 2/4: Carregando modelo de embeddings...
2026-06-21 17:00:07 [INFO]   Embeddings carregado: bert-base-portuguese-cased
2026-06-21 17:00:07 [INFO] Etapa 3/4: Carregando e indexando bulas...
2026-06-21 17:00:12 [INFO]   FAISS: 5234 vetores indexados
2026-06-21 17:00:12 [INFO] Etapa 4/4: Inicializando provedor de linguagem...
2026-06-21 17:00:18 [INFO]   Backend ativo: direta
2026-06-21 17:00:18 [INFO] Pipeline inicializado em 17.0 segundos
2026-06-21 17:00:18 [INFO] ======================================================================
2026-06-21 17:00:18 [INFO] ------------------------------------------------------------
2026-06-21 17:00:18 [INFO] Consulta #1: 'Posso tomar Amoxicilina com Metotrexato?'
2026-06-21 17:00:18 [INFO]   Etapa 1/3: Reconhecendo entidades...
2026-06-21 17:00:18 [INFO]   Medicamentos encontrados: ['amoxicilina', 'metotrexato']
2026-06-21 17:00:18 [INFO]   Etapa 2/3: Buscando e classificando interações...
2026-06-21 17:00:18 [INFO]     Par amoxicilina + metotrexato: 2 trechos relevantes (top: 0.892)
2026-06-21 17:00:21 [INFO]     GPT4All: classe=2 (GRAVE_CONTRAINDICADA)
2026-06-21 17:00:21 [INFO]   Etapa 3/3: Concluído em 3200ms. 1 interações encontradas.
2026-06-21 17:00:21 [INFO] ------------------------------------------------------------
2026-06-21 17:00:21 [INFO] Consulta #2: 'Dipirona e AAS juntos têm problema?'
...
```

---

## 7. Commits Atômicos

### Commit 1: Funções de carregamento e sanitização
```
feat: Sprint 5 — carregar_trechos_bulas + sanitizar_consulta + logging
```

### Commit 2: PipelineInteracao — inicializar() e _analisar_par()
```
feat: Sprint 5 — PipelineInteracao: inicializar modelos, FAISS, GPT4All
```

### Commit 3: PipelineInteracao — consultar()
```
feat: Sprint 5 — PipelineInteracao.consultar: NER → FAISS → prompt → GPT4All → JSON
```

### Commit 4: 8 consultas de demonstração
```
feat: Sprint 5 — 8 consultas demo: grave, leve, sem, não encontrado, múltiplos, ambígua
```

### Commit 5: Comparação com/sem contexto + chunking
```
feat: Sprint 5 — comparar_com_sem_contexto + 3 estratégias de chunking
```

### Commit 6: Segurança, falhas, limitações, conclusão
```
feat: Sprint 5 — injeção de prompt, análise de falhas, riscos, limitações
```
