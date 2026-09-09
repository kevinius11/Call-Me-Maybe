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
        "You are an expert API router. Analyze the user's request and select "
        "the most appropriate function from the available function definitions. "
        "Extract the values required by that function from the user's request. "
        "For replacement parameters, use the exact symbol or text requested by "
        "the user; descriptive words such as 'asterisks' refer to the symbol '*'. "
        "For regex parameters, match the target text itself without adding "
        "surrounding text or combining separate occurrences. "
        "Return the result using exactly this JSON structure: "
        '{"fn_name": "<function_name>", "args": {<parameter_name>: <value>}}. '
        "The function name must match one of the available functions. "
        "The args object must contain the parameters required by that function. "
        "Do not use any other field names. Do not use 'fn_params'. "
        "Do not add explanations or Markdown."
        "For replacement parameters, use the actual replacement"
        "symbol when the user refers to a symbol by its name."
    )

    # Formateamos funciones
    functions_text = ""
    for function in functions:
        functions_text += f"{function.name}: {function.description}\n"
        for param_name, param_def in function.parameters.items():
            functions_text += f"  - {param_name}: {param_def.type}\n"

    # Request del usuario
    user_request = f"User request: {prompt}"

    return "\n\n".join([instruction, functions_text, user_request])
