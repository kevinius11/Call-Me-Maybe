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
        "You are an expert API router. Given the user's request"
        " and the list of JSON function schemas, select the most accurate"
        " function to resolve the request and return the selected function"
        " name and its required parameters strictly in JSON format."
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
