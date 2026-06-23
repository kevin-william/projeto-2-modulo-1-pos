# Sprint 2 — Notebook 02 (Engenharia de Prompt com GPT4All)

**Objetivo:** Implementar o Notebook 02 demonstrando 3 técnicas de engenharia de
prompt (zero-shot, few-shot, cadeia de pensamento) usando GPT4All como LLM local,
com parsing robusto de JSON, avaliação quantitativa e proteção contra injeção.

**Duração:** 3-4 horas  
**Commits:** 5 atômicos  
**Rubricas cobertas:** Rubrica 2 (5 itens)

---

## 1. Arquitetura do LLM (Classe ProvedorLinguagem)

### 1.1 Estratégia de fallback em três camadas

```
┌─────────────────────────────────────────┐
│         ProvedorLinguagem               │
│                                          │
│  CAMADA 1: GPT4All Ligação Direta       │
│  from gpt4all import GPT4All            │
│  modelo = GPT4All("Llama-3-8B.gguf")    │
│  modelo.generate(consulta)              │
│         ↓ se falhar                      │
│  CAMADA 2: GPT4All Servidor API         │
│  cliente = OpenAI(base_url=localhost)    │
│  cliente.chat.completions.create()       │
│         ↓ se falhar                      │
│  CAMADA 3: Heurística de PalavrasChave  │
│  classificação instantânea sem LLM      │
└─────────────────────────────────────────┘
```

### 1.2 Implementação completa da classe

```python
class ProvedorLinguagem:
    """
    Provedor de modelo de linguagem com degradação progressiva em três camadas.

    Camada 1 — Ligação Direta:
        Usa o binding Python oficial do GPT4All. O modelo .gguf é carregado
        integralmente em memória RAM. Inferência via CPU.
        Requer: pip install gpt4all

    Camada 2 — Servidor API:
        Conecta ao aplicativo desktop do GPT4All atuando como servidor HTTP
        local na porta 4891. Interface compatível com OpenAI.
        Requer: GPT4All desktop app com "Enable Local API Server" ativado

    Camada 3 — Heurística:
        Classificação baseada exclusivamente em palavras-chave.
        Não requer modelo de linguagem. Instantânea.
        Usada apenas quando as duas camadas anteriores falham.

    Atributos:
        nome_modelo (str): Nome do arquivo .gguf (ex: "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        modelo_direto (GPT4All | None): Instância do modelo via binding direto
        cliente_api (OpenAI | None): Cliente HTTP para o servidor API local
        camada_ativa (str): Qual camada está em uso ("direta", "api", "heuristica")

    Exemplo:
        >>> provedor = ProvedorLinguagem()
        >>> resposta = provedor.gerar("Classifique a interação entre Amoxicilina e Ibuprofeno.")
        >>> print(resposta)
    """

    def __init__(self, nome_modelo="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
        self.nome_modelo = nome_modelo
        self.modelo_direto = None
        self.cliente_api = None
        self.camada_ativa = None
        self._inicializar_camada()

    def _inicializar_camada(self):
        """Tenta ativar as camadas em ordem: direta → api → heurística."""
        # Camada 1: Ligação Direta
        try:
            from gpt4all import GPT4All
            registro.info("Tentando camada 1: ligação direta GPT4All...")
            self.modelo_direto = GPT4All(self.nome_modelo)
            self.camada_ativa = "direta"
            registro.info("✅ Camada 1 ativa: GPT4All ligação direta (%s)", self.nome_modelo)
            return
        except Exception as erro:
            registro.warning("❌ Camada 1 falhou: %s", erro)

        # Camada 2: Servidor API
        try:
            from openai import OpenAI
            registro.info("Tentando camada 2: servidor API GPT4All...")
            self.cliente_api = OpenAI(
                base_url="http://localhost:4891/v1",
                api_key="gpt4all"
            )
            # Testa a conexão
            self.cliente_api.models.list()
            self.camada_ativa = "api"
            registro.info("✅ Camada 2 ativa: servidor API GPT4All (localhost:4891)")
            return
        except Exception as erro:
            registro.warning("❌ Camada 2 falhou: %s", erro)

        # Camada 3: Heurística (fallback final)
        self.camada_ativa = "heuristica"
        registro.warning("⚠️  Camada 3 ativa: heurística de palavras-chave (sem LLM)")

    def gerar(self, consulta, maximo_tokens=200):
        """
        Gera uma resposta usando a camada ativa.

        Argumentos:
            consulta (str): O prompt completo a ser enviado ao modelo.
            maximo_tokens (int): Limite de tokens na resposta.

        Retorna:
            str: Texto gerado pelo modelo.
        """
        registro.info(
            "Gerando resposta | camada=%s | consulta=%d caracteres | max_tokens=%d",
            self.camada_ativa, len(consulta), maximo_tokens
        )
        tempo_inicio = __import__("time").time()

        if self.camada_ativa == "direta":
            resposta = self.modelo_direto.generate(consulta, max_tokens=maximo_tokens)
        elif self.camada_ativa == "api":
            resposta_api = self.cliente_api.chat.completions.create(
                model=self.nome_modelo,
                messages=[{"role": "user", "content": consulta}],
                max_tokens=maximo_tokens,
                temperature=0.1,
            )
            resposta = resposta_api.choices[0].message.content
        else:
            resposta = self._classificar_por_palavras_chave(consulta)

        tempo_total = (__import__("time").time() - tempo_inicio) * 1000
        registro.info(
            "Resposta gerada | camada=%s | tempo=%.0fms | tamanho=%d caracteres",
            self.camada_ativa, tempo_total, len(resposta)
        )
        return resposta

    def _classificar_por_palavras_chave(self, texto):
        """
        Classificação heurística de fallback.

        Varre o texto em busca de palavras-chave de cada classe.
        Prioridade: GRAVE > LEVE > SEM_INTERAÇÃO.
        """
        texto_minusculo = texto.lower()

        # Verifica classe 2 (GRAVE) — maior prioridade
        for palavra in ["contraindicado", "fatal", "risco de morte",
                        "não administrar", "rabdomiólise"]:
            if palavra in texto_minusculo:
                return '{"classe": 2, "justificativa": "Palavra-chave grave detectada: '
                       f'{palavra}", "evidencia": "Classificação heurística (fallback)"}}'

        # Verifica classe 0 (SEM_INTERAÇÃO)
        for frase in ["não há interação", "sem interação", "não foram observadas",
                      "é seguro", "pode ser usado"]:
            if frase in texto_minusculo:
                return '{"classe": 0, "justificativa": "Frase de ausência de interação '
                       f'detectada: {frase}", "evidencia": "Classificação heurística (fallback)"}}'

        # Verifica classe 1 (LEVE_MODERADA)
        for palavra in ["monitorar", "ajustar", "cautela", "precaução",
                        "pode aumentar", "pode reduzir"]:
            if palavra in texto_minusculo:
                return '{"classe": 1, "justificativa": "Palavra-chave de interação leve '
                       f'detectada: {palavra}", "evidencia": "Classificação heurística (fallback)"}}'

        # Padrão: sem interação
        return '{"classe": 0, "justificativa": "Nenhuma palavra-chave detectada", '
               '"evidencia": "Classificação heurística (fallback)"}'
```

---

## 2. Dados de Teste (30 Pares com Gabarito)

### 2.1 Estrutura de cada par

```python
{
    "medicamento_principal": str,   # Medicamento dono da bula
    "medicamento_secundario": str,  # Medicamento com o qual interage
    "trecho_bula": str,             # Contexto extraído da bula real
    "classe_esperada": int,         # 0 (SEM), 1 (LEVE), 2 (GRAVE)
}
```

### 2.2 Os 30 pares completos

```python
pares_para_teste = [
    # ═══════════ CLASSE 0: SEM INTERAÇÃO (10 pares) ═══════════
    {
        "medicamento_principal": "amoxicilina",
        "medicamento_secundario": "paracetamol",
        "trecho_bula": (
            "Não há interações clinicamente relevantes com paracetamol "
            "quando utilizado nas doses recomendadas."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "atorvastatina",
        "medicamento_secundario": "insulina",
        "trecho_bula": (
            "Não foram observadas interações clinicamente significativas "
            "entre atorvastatina e insulina."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "alopurinol",
        "medicamento_secundario": "paracetamol",
        "trecho_bula": (
            "Não há interações conhecidas entre alopurinol e paracetamol. "
            "O uso concomitante é considerado seguro."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "captopril",
        "medicamento_secundario": "amoxicilina",
        "trecho_bula": (
            "Não existem relatos de interação entre captopril e amoxicilina. "
            "Ambos podem ser administrados simultaneamente sem risco."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "sinvastatina",
        "medicamento_secundario": "omeprazol",
        "trecho_bula": (
            "Estudos clínicos não demonstraram interação clinicamente "
            "relevante entre sinvastatina e omeprazol."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "amoxicilina",
        "medicamento_secundario": "dipirona",
        "trecho_bula": (
            "A dipirona pode ser administrada concomitantemente com "
            "amoxicilina sem risco de interação medicamentosa."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "atorvastatina",
        "medicamento_secundario": "losartana",
        "trecho_bula": (
            "Não há evidência de interação medicamentosa entre "
            "atorvastatina e losartana nas doses terapêuticas habituais."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "alopurinol",
        "medicamento_secundario": "prednisona",
        "trecho_bula": (
            "O alopurinol não apresenta interação com corticosteroides "
            "como a prednisona."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "captopril",
        "medicamento_secundario": "metformina",
        "trecho_bula": (
            "Não há interação descrita entre captopril e metformina "
            "nas bulas consultadas. O uso concomitante é seguro."
        ),
        "classe_esperada": 0,
    },
    {
        "medicamento_principal": "sinvastatina",
        "medicamento_secundario": "levotiroxina",
        "trecho_bula": (
            "A sinvastatina pode ser usada com segurança junto à "
            "levotiroxina, sem interações relatadas na literatura."
        ),
        "classe_esperada": 0,
    },

    # ═══════════ CLASSE 1: LEVE MODERADA (10 pares) ═══════════
    {
        "medicamento_principal": "amoxicilina",
        "medicamento_secundario": "probenecida",
        "trecho_bula": (
            "A probenecida reduz a secreção tubular renal da amoxicilina. "
            "No uso concomitante, pode haver aumento dos níveis de "
            "amoxicilina no sangue e no prolongamento dessa alteração."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "amoxicilina",
        "medicamento_secundario": "varfarina",
        "trecho_bula": (
            "Existem casos raros de INR aumentada em pacientes mantidos "
            "com varfarina, ao receberem um curso de tratamento com "
            "amoxicilina. Se a coadministração é necessária, o tempo de "
            "protrombina deve ser cuidadosamente monitorado."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "amoxicilina",
        "medicamento_secundario": "alopurinol",
        "trecho_bula": (
            "A administração concomitante de alopurinol durante o "
            "tratamento com amoxicilina pode aumentar a probabilidade "
            "de reações alérgicas da pele."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "atorvastatina",
        "medicamento_secundario": "ciclosporina",
        "trecho_bula": (
            "Miopatia devido à lesão dos músculos pode ocorrer em "
            "pacientes que usam atorvastatina, sendo mais frequente "
            "naqueles que usam também ciclosporina."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "atorvastatina",
        "medicamento_secundario": "eritromicina",
        "trecho_bula": (
            "A administração concomitante de atorvastatina com "
            "medicamentos inibidores do citocromo P450 como eritromicina "
            "e claritromicina pode alterar a concentração plasmática "
            "da atorvastatina."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "sinvastatina",
        "medicamento_secundario": "varfarina",
        "trecho_bula": (
            "É importante informar ao seu médico se estiver tomando "
            "anticoagulantes como varfarina. A sinvastatina pode "
            "potencializar o efeito anticoagulante, exigindo "
            "monitoramento mais frequente do INR."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "alopurinol",
        "medicamento_secundario": "captopril",
        "trecho_bula": (
            "Um risco aumentado de hipersensibilidade foi relatado quando "
            "o alopurinol é administrado com inibidores da enzima "
            "conversora de angiotensina como o captopril, especialmente "
            "em pacientes com insuficiência renal. Recomenda-se cautela."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "captopril",
        "medicamento_secundario": "ibuprofeno",
        "trecho_bula": (
            "Os anti-inflamatórios não esteroidais como o ibuprofeno "
            "podem reduzir o efeito anti-hipertensivo do captopril. "
            "Recomenda-se monitoramento da pressão arterial durante "
            "o uso concomitante."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "sinvastatina",
        "medicamento_secundario": "diltiazem",
        "trecho_bula": (
            "O uso concomitante de sinvastatina com diltiazem pode "
            "aumentar os níveis séricos da sinvastatina. Recomenda-se "
            "ajuste de dose e monitoramento de efeitos musculares."
        ),
        "classe_esperada": 1,
    },
    {
        "medicamento_principal": "alopurinol",
        "medicamento_secundario": "hidroclorotiazida",
        "trecho_bula": (
            "A hidroclorotiazida pode reduzir a eficácia do alopurinol. "
            "Recomenda-se monitoramento dos níveis de ácido úrico e "
            "ajuste de dose se necessário."
        ),
        "classe_esperada": 1,
    },

    # ═══════════ CLASSE 2: GRAVE CONTRAINDICADA (10 pares) ═══════════
    {
        "medicamento_principal": "sinvastatina",
        "medicamento_secundario": "itraconazol",
        "trecho_bula": (
            "É muito importante informar ao seu médico se você for tomar "
            "sinvastatina associado a agentes antifúngicos como o "
            "itraconazol, cetoconazol, posaconazol ou voriconazol, pois "
            "o risco de problemas musculares nessa situação é maior. "
            "Em raras ocasiões, problemas musculares podem ser graves, "
            "incluindo rompimento muscular resultando em dano renal que "
            "pode ser fatal."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "amoxicilina",
        "medicamento_secundario": "metotrexato",
        "trecho_bula": (
            "O uso concomitante de amoxicilina com metotrexato é "
            "contraindicado devido ao risco de toxicidade grave. A "
            "amoxicilina reduz a secreção tubular do metotrexato, "
            "podendo causar níveis tóxicos e risco de morte."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "atorvastatina",
        "medicamento_secundario": "amiodarona",
        "trecho_bula": (
            "O uso de atorvastatina com amiodarona é contraindicado. "
            "Esta combinação aumenta significativamente o risco de "
            "rabdomiólise, que pode levar a insuficiência renal aguda "
            "e morte."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "captopril",
        "medicamento_secundario": "alopurinol",
        "trecho_bula": (
            "Reações de hipersensibilidade graves, incluindo síndrome "
            "de Stevens-Johnson, foram relatadas com o uso concomitante "
            "de captopril e alopurinol. Esta combinação é contraindicada."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "sinvastatina",
        "medicamento_secundario": "cetoconazol",
        "trecho_bula": (
            "O cetoconazol é contraindicado com sinvastatina. O risco "
            "de miopatia grave e rabdomiólise é extremamente elevado, "
            "podendo ser fatal. Não administrar esta combinação."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "alopurinol",
        "medicamento_secundario": "azatioprina",
        "trecho_bula": (
            "A combinação de alopurinol com azatioprina é contraindicada. "
            "O alopurinol inibe o metabolismo da azatioprina, podendo "
            "causar toxicidade grave da medula óssea com risco de vida."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "amoxicilina",
        "medicamento_secundario": "alopurinol_grave",
        "trecho_bula": (
            "Em pacientes com histórico de hipersensibilidade, a "
            "administração de amoxicilina pode desencadear reações "
            "alérgicas graves e ocasionalmente fatais, incluindo "
            "anafilaxia e síndrome de Stevens-Johnson."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "atorvastatina",
        "medicamento_secundario": "saquinavir",
        "trecho_bula": (
            "O uso concomitante de atorvastatina com inibidores da "
            "protease do HIV como saquinavir é contraindicado. O risco "
            "de rabdomiólise fatal é inaceitável. Não utilizar esta "
            "combinação."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "captopril",
        "medicamento_secundario": "suplemento_potassio",
        "trecho_bula": (
            "A administração de suplementos de potássio com captopril "
            "pode causar hipercalemia grave e potencialmente fatal. Esta "
            "combinação é contraindicada, especialmente em pacientes com "
            "insuficiência renal."
        ),
        "classe_esperada": 2,
    },
    {
        "medicamento_principal": "sinvastatina",
        "medicamento_secundario": "genfibrozila",
        "trecho_bula": (
            "A combinação de sinvastatina com genfibrozila é "
            "contraindicada. O risco de rabdomiólise é multiplicado "
            "por dez nesta combinação. Casos de morte por insuficiência "
            "renal aguda foram relatados na literatura médica."
        ),
        "classe_esperada": 2,
    },
]
```

---

## 3. Os Três Templates de Prompt

### 3.1 Template Base (usado por zero-shot e few-shot)

```python
modelo_prompt_base = """[PAPEL PROFISSIONAL]
Você é um farmacêutico clínico especializado em interações medicamentosas
com vinte anos de experiência em segurança do paciente. Você trabalha no
Hospital das Clínicas revisando prescrições para evitar interações perigosas.

[TAREFA]
Analise o trecho de bula abaixo e classifique a interação entre os dois
medicamentos mencionados.

[CLASSES POSSÍVEIS]
0 = SEM_INTERACAO: não há interação clinicamente relevante ou o uso
    concomitante é considerado seguro
1 = LEVE_MODERADA: existe interação que requer monitoramento, ajuste de
    dose ou precaução, mas não há contraindicação absoluta
2 = GRAVE_CONTRAINDICADA: a interação apresenta risco de evento adverso
    grave, contraindicação absoluta ou risco de morte

[TRECHO DA BULA]
{trecho_bula}

[MEDICAMENTOS ANALISADOS]
Medicamento principal (dono da bula): {medicamento_principal}
Medicamento secundário (mencionado no trecho): {medicamento_secundario}

[FORMATO DE SAÍDA — OBRIGATÓRIO]
Responda EXCLUSIVAMENTE com um objeto JSON válido, sem nenhum texto antes
ou depois. O JSON deve conter exatamente estes três campos:

{{"classe": <0, 1 ou 2>, "justificativa": "<breve justificativa da classificação>", "evidencia": "<trecho exato da bula que fundamenta a classificação>"}}

Não inclua comentários, explicações ou formatação adicional."""
```

### 3.2 Template Few-Shot (adiciona 3 exemplos)

```python
exemplos_few_shot = """
[EXEMPLOS DE CLASSIFICAÇÃO]
Estes três exemplos mostram exatamente como classificar:

EXEMPLO 1 — SEM INTERAÇÃO:
Trecho: "Não há interações clinicamente relevantes com paracetamol quando utilizado nas doses recomendadas."
Medicamentos: amoxicilina + paracetamol
Resposta: {{"classe": 0, "justificativa": "A bula afirma explicitamente a ausência de interação com paracetamol", "evidencia": "Não há interações clinicamente relevantes com paracetamol quando utilizado nas doses recomendadas"}}

EXEMPLO 2 — LEVE MODERADA:
Trecho: "A probenecida reduz a secreção tubular renal da amoxicilina. No uso concomitante, pode haver aumento dos níveis de amoxicilina no sangue."
Medicamentos: amoxicilina + probenecida
Resposta: {{"classe": 1, "justificativa": "Interação farmacocinética que requer monitoramento dos níveis séricos, sem contraindicação absoluta", "evidencia": "A probenecida reduz a secreção tubular renal da amoxicilina"}}

EXEMPLO 3 — GRAVE CONTRAINDICADA:
Trecho: "O uso concomitante de amoxicilina com metotrexato é contraindicado devido ao risco de toxicidade grave e potencialmente fatal."
Medicamentos: amoxicilina + metotrexato
Resposta: {{"classe": 2, "justificativa": "Contraindicação explícita com risco de morte descrito na bula", "evidencia": "O uso concomitante de amoxicilina com metotrexato é contraindicado devido ao risco de toxicidade grave e potencialmente fatal"}}

Agora classifique o caso abaixo seguindo o mesmo padrão.
"""

modelo_prompt_few_shot = modelo_prompt_base.replace(
    "[FORMATO DE SAÍDA — OBRIGATÓRIO]",
    exemplos_few_shot + "\n[FORMATO DE SAÍDA — OBRIGATÓRIO]"
)
```

### 3.3 Template Cadeia de Pensamento (Chain-of-Thought)

```python
modelo_prompt_cadeia_pensamento = """[PAPEL PROFISSIONAL]
Você é um farmacêutico clínico especializado em interações medicamentosas
com vinte anos de experiência em segurança do paciente.

[TAREFA]
Classifique a interação entre dois medicamentos seguindo rigorosamente
as três etapas de raciocínio abaixo. Você deve executar cada etapa
mentalmente antes de prosseguir para a próxima.

[RACIOCÍNIO PASSO A PASSO]
Etapa 1 — IDENTIFICAÇÃO:
O trecho da bula menciona alguma interação entre os medicamentos?
Se o trecho apenas lista nomes sem descrever efeito, responda que não
há informação suficiente.

Etapa 2 — AVALIAÇÃO DE GRAVIDADE:
Examine as palavras usadas no trecho para determinar a gravidade:
• PALAVRAS GRAVES (indicam CLASSE 2): "contraindicado", "fatal",
  "risco de morte", "não administrar", "nunca associar", "rabdomiólise",
  "Stevens-Johnson", "hemorragia grave", "insuficiência renal aguda"
• PALAVRAS LEVES (indicam CLASSE 1): "monitorar", "ajustar dose",
  "cautela", "precaução", "pode aumentar", "pode reduzir",
  "recomenda-se", "potencializa"
• FRASES DE AUSÊNCIA (indicam CLASSE 0): "não há interação",
  "sem interação", "não foram observadas", "é seguro", "pode ser usado"

Etapa 3 — CLASSIFICAÇÃO FINAL:
Com base nas etapas 1 e 2, atribua a classe 0, 1 ou 2.
Palavras graves têm precedência sobre palavras leves.
Palavras leves têm precedência sobre frases de ausência.

[CLASSES POSSÍVEIS]
0 = SEM_INTERACAO  1 = LEVE_MODERADA  2 = GRAVE_CONTRAINDICADA

[TRECHO DA BULA]
{trecho_bula}

[MEDICAMENTOS ANALISADOS]
Medicamento principal: {medicamento_principal}
Medicamento secundário: {medicamento_secundario}

[FORMATO DE SAÍDA — OBRIGATÓRIO]
Responda EXCLUSIVAMENTE com um objeto JSON:
{{"classe": <0, 1 ou 2>, "justificativa": "<breve>", "evidencia": "<trecho exato>"}}"""
```

---

## 4. Parsing Robusto de JSON

```python
def analisar_resposta_json(resposta_bruta):
    """
    Converte a resposta textual do LLM em um dicionário estruturado.

    Estratégias de extração (tentadas em ordem):
    1. Interpretar a string inteira como JSON via json.loads()
    2. Remover marcadores markdown (```json ... ```) e tentar novamente
    3. Extrair campos individuais via expressões regulares (fallback)

    Argumentos:
        resposta_bruta (str | None): Resposta textual do LLM.

    Retorna:
        dict | None: Dicionário com 'classe', 'justificativa', 'evidencia'.
                     Retorna None se todas as estratégias falharem.
    """
    if resposta_bruta is None:
        registro.warning("analisar_resposta_json: resposta_bruta é None")
        return None

    resposta_limpa = resposta_bruta.strip()
    registro.info("Analisando resposta JSON (%d caracteres)", len(resposta_limpa))

    # Estratégia 1: JSON direto
    try:
        dados = json.loads(resposta_limpa)
        if _validar_estrutura_json(dados):
            registro.info("✅ Estratégia 1: JSON direto válido")
            return dados
    except json.JSONDecodeError:
        pass

    # Estratégia 2: Remover markdown e tentar novamente
    texto_sem_markdown = re.sub(
        r"```(?:json)?\s*|\s*```", "", resposta_limpa
    ).strip()
    try:
        dados = json.loads(texto_sem_markdown)
        if _validar_estrutura_json(dados):
            registro.info("✅ Estratégia 2: JSON após remover markdown")
            return dados
    except json.JSONDecodeError:
        pass

    # Estratégia 3: Regex fallback
    registro.warning("⚠️  Estratégias 1 e 2 falharam, usando regex fallback")
    classe_encontrada = re.search(r'"classe"\s*:\s*(\d)', resposta_limpa)
    if classe_encontrada:
        classe = int(classe_encontrada.group(1))
        if classe in (0, 1, 2):
            justificativa_encontrada = re.search(
                r'"justificativa"\s*:\s*"([^"]*)"', resposta_limpa
            )
            evidencia_encontrada = re.search(
                r'"evidencia"\s*:\s*"([^"]*)"', resposta_limpa
            )
            dados = {
                "classe": classe,
                "justificativa": (
                    justificativa_encontrada.group(1)
                    if justificativa_encontrada else ""
                ),
                "evidencia": (
                    evidencia_encontrada.group(1)
                    if evidencia_encontrada else ""
                ),
                "_metodo_extracao": "regex_fallback",
            }
            registro.info("✅ Estratégia 3: Regex extraiu classe=%d", classe)
            return dados

    registro.error("❌ Todas as estratégias de parsing falharam")
    return None


def _validar_estrutura_json(dados):
    """
    Verifica se o dicionário contém o campo 'classe' com valor 0, 1 ou 2.

    Retorna:
        bool: True se a estrutura é válida.
    """
    return (
        isinstance(dados, dict)
        and "classe" in dados
        and isinstance(dados["classe"], int)
        and dados["classe"] in (0, 1, 2)
    )
```

---

## 5. Estrutura do Caderno (11 células)

| Célula | Tipo | Conteúdo |
|---|---|---|
| 1 | Markdown | Título, objetivo, rubrica, fluxo do caderno |
| 2 | Code | **Configuração:** logging, classe `ProvedorLinguagem`, 30 pares de teste. `registro.info("Caderno 02 iniciado. Pares carregados: %d", len(pares_para_teste))` |
| 3 | Markdown | Explicação do template base: papel, tarefa, classes, formato JSON |
| 4 | Code + Markdown | **Zero-shot:** executar 30 pares, coletar métricas. `registro.info("Zero-shot: %d pares, acurácia=%.2f")` |
| 5 | Code + Markdown | **Few-shot:** mesmo prompt + 3 exemplos. Métricas. |
| 6 | Code + Markdown | **Cadeia de pensamento:** prompt com 3 etapas. Métricas. |
| 7 | Code | `analisar_resposta_json()` — demonstração com 5 casos de teste |
| 8 | Markdown | **Tabela comparativa:** acurácia, F1 por classe, JSON válido, latência, backend |
| 9 | Code + Markdown | **Injeção de prompt:** ataque + sanitização |
| 10 | Markdown | Conclusão: qual técnica usar no caderno 05 |
| 11 | Code | `registro.info("Caderno 02 concluído. Backend: %s", provedor.camada_ativa)` |

---

## 6. Commits Atômicos

### Commit 1: Classe ProvedorLinguagem + pares de teste
```
feat: Sprint 2 — ProvedorLinguagem com 3 camadas de fallback + 30 pares de teste
```
Arquivos: `c02_engenharia_prompt.ipynb` (células 1-2)

### Commit 2: Templates de prompt (zero-shot, few-shot, CoT)
```
feat: Sprint 2 — templates de prompt: base, few-shot com 3 exemplos, cadeia de pensamento
```
Arquivos: `c02_engenharia_prompt.ipynb` (células 3-4)

### Commit 3: Execução zero-shot + few-shot
```
feat: Sprint 2 — execução zero-shot e few-shot nos 30 pares com métricas
```
Arquivos: `c02_engenharia_prompt.ipynb` (células 4-5)

### Commit 4: Cadeia de pensamento + parsing JSON
```
feat: Sprint 2 — cadeia de pensamento + analisar_resposta_json com 3 estratégias
```
Arquivos: `c02_engenharia_prompt.ipynb` (células 6-7)

### Commit 5: Avaliação comparativa + injeção de prompt + conclusão
```
feat: Sprint 2 — tabela comparativa, injeção de prompt, sanitização, conclusão
```
Arquivos: `c02_engenharia_prompt.ipynb` (células 8-11)
