from src.schemas import FunctionDefinition


def build_prompt(
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """
    Construye el prompt estructurado para el LLM encargado de la selección
    de funciones.

    Args:
        prompt: La consulta o petición original del usuario.
        functions: Lista de definiciones de funciones disponibles.

    Returns:
        String con el prompt final estructurado para el LLM.
    """

    instruction = (
        "You are an expert API router. Analyze the user's request and select "
        "the most appropriate function from the available function "
        "definitions. "
        "Extract the values required by that function from the user's "
        "request. "
        "Copy explicitly provided parameter values exactly as they appear in "
        "the request. "
        "When a value is explicitly enclosed in single or double quotes in "
        "the user request, treat the complete quoted text as that parameter "
        "value and stop the value at the closing quote. "
        "For replacement parameters, use the exact symbol or text requested "
        "by the user; descriptive words such as 'asterisks' refer to the "
        "symbol '*'. "
        "For regex parameters, match the target text itself without adding "
        "surrounding text or combining separate occurrences. "
        "Return the result using exactly this JSON structure: "
        '{"name": "<function_name>", "parameters": '
        '{<parameter_name>: <value>}}. '
        "The function name must match one of the available functions. "
        "The parameters object must contain the parameters required by that "
        "function. "
        "Do not use any other field names. Do not use 'fn_params'. "
        "Do not add explanations or Markdown."
    )

    functions_text = ""

    for function in functions:
        functions_text += (
            f"{function.name}: {function.description}\n"
        )

        for param_name, param_def in function.parameters.items():
            functions_text += (
                f"  - {param_name}: {param_def.type}\n"
            )

    user_request = f"User request: {prompt}"

    return "\n\n".join([
        instruction,
        functions_text,
        user_request,
    ])
