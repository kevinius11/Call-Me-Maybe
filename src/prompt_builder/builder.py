from src.schemas import FunctionDefinition


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """
    Construye el prompt estructurado para el LLM encargado de la selección
    de funciones.

    Args:
        prompt: La consulta o petición original del usuario.
        functions: Lista de definiciones de funciones disponibles.

    Returns:
        String con el prompt final estructurado para el LLM.
    """

    # Instruccion general al LLM

    instruction = (
        "You are an API router. "
        "Select exactly one function that matches the user request and extract "
        "the required arguments. "

        "Use only information that comes from USER REQUEST. "
        "Never copy argument values from these instructions or from the function "
        "definitions. "
        "Do not invent, complete, paraphrase, summarize, translate, normalize, "
        "or extend argument values. "

        "For string arguments, preserve the exact user-provided text. "
        "Preserve spaces, punctuation, braces, quotation marks, apostrophes, "
        "slashes, backslashes, and other special characters. "

        "If the user explicitly gives a quoted value, use the text inside the "
        "outer quotation marks as the value. "
        "Quotation marks that belong to the value must be preserved. "

        "For template arguments, copy the complete template text exactly as "
        "provided by the user. Never add placeholders, examples, explanations, "
        "or additional sentences. "

        "For file paths, preserve every character exactly, including leading "
        "slashes and Windows backslashes. "

        "For SQL arguments, copy the complete query exactly as provided. "
        "Never append or continue the query. "

        "For replacement arguments, use the exact replacement requested. "
        "The word 'asterisks' means the symbol '*'. "

        "For regex arguments, match the requested target text directly. "

        "Return only a JSON object containing the selected function name and "
        "its required arguments. Do not return explanations or Markdown."
    )

    # Formateamos funciones
    functions_text = ""
    for function in functions:
        functions_text += f"{function.name}: {function.description}\n"
        for param_name, param_def in function.parameters.items():
            functions_text += f"  - {param_name}: {param_def.type}\n"

    # Request del usuario
    user_request = f"User request: {prompt}"

    return "\n\n".join([
        instruction,
        functions_text,
        user_request
        ])
