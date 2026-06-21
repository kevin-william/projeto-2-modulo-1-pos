"""Append cell 2.3 (sentiment-analysis) to notebook."""
import nbformat as nbf

nb = nbf.read("C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb", as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell(
    "## 2.3 Pipeline: sentiment-analysis em Frases Clinicas\n\n"
    "Usando o pipeline default de sentiment-analysis do Hugging Face para "
    "classificar frases extraidas de bulas medicas. O objetivo nao e obter "
    "resultados perfeitos, mas **demonstrar as limitacoes** de um modelo "
    "generico em dominio especializado — o que motiva o fine-tuning posterior."
))

nb.cells.append(nbf.v4.new_code_cell(
    "classifier = pipeline(\"sentiment-analysis\")\n\n"
    "# Frases reais de bulas medicas\n"
    "frases = [\n"
    '    "O uso concomitante e contraindicado devido ao risco de arritmia fatal.",\n'
    '    "Nao ha interacoes conhecidas com este medicamento.",\n'
    '    "Recomenda-se monitoramento da funcao renal durante o tratamento.",\n'
    '    "A administracao concomitante de Amoxicilina com Metotrexato pode aumentar a toxicidade.",\n'
    '    "O medicamento e seguro e bem tolerado pela maioria dos pacientes.",\n'
    "]\n\n"
    "for frase in frases:\n"
    "    resultado = classifier(frase)[0]\n"
    '    print(f"[{resultado[\"label\"]:>8} | {resultado[\"score\"]:.3f}] {frase}\")'
))

nb.cells.append(nbf.v4.new_markdown_cell(
    "**Analise:**\n\n"
    "- O modelo generico classifica frases como POSITIVE/NEGATIVE com base "
    "no tom emocional, **nao no significado clinico**\n"
    "- Frase com \"contraindicado\" e \"fatal\" → NEGATIVE (correto por acaso)\n"
    "- Frase com \"nao ha interacoes\" → POSITIVE (correto por acaso)\n"
    "- **Limitacao:** \"aumentar a toxicidade\" pode ser classificado como "
    "NEGATIVE pelo tom, mas o modelo nao entende que isso descreve uma "
    "**interacao medicamentosa grave**\n"
    "- **Conclusao:** Precisamos de modelos treinados em dominio clinico "
    "(como o `clinicalnerpt-chemical`) e fine-tuning especifico para "
    "classificacao de interacoes"
))

nbf.write(nb, "C:/workspace/python/projeto-2-modulo-1-pos/c01_modelos_llm.ipynb")
print("Cell 2.3 appended.")
