import json


def load_function_definitions(path: str) -> list[dict]:
    """
    Carga el JSON con las definiciones de funciones del LLM

    Args:
        path: Ruta hacia el JSON con las definiciones de las funciones.

    Returns:
        Lista de tipo diccionario con las definiciones de funciones.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        json.JSONDecodeError: Si el archivo no cumple con las reglas
        de sintaxis.
        ValueError: En caso de que la estructura del JSON no sea una lista.
    """
    try:
        with open(path) as f:
            datas: list[dict] = json.load(f)
            if not isinstance(datas, list):
                raise ValueError("JSON must be a list, "
                                 "not object/dictionaries.")
            if not all(isinstance(data, dict) for data in datas):
                raise ValueError("All definitions must be dict type.")
            return datas
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Functions definitions file not found: {path}"
            )


def load_prompts(path: str) -> list[str]:
    """
    Carga los prompts desde un archivo JSON.

    Args:
        path: Ruta al archivo JSON de prompts.

    Returns:
        Lista de strings con los prompts.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el JSON es invalido o tiene una estructura incorrecta.
    """
    try:
        with open(path) as f:
            data_prompts = json.load(f)

            if not isinstance(data_prompts, list):
                raise ValueError(
                    "JSON must be a list, not object/dictionaries."
                )

            prompts = []

            for item in data_prompts:
                if not isinstance(item, dict):
                    raise ValueError(
                        "All prompt entries must be dictionaries."
                    )

                prompt = item.get("prompt")

                if not isinstance(prompt, str):
                    raise ValueError(
                        "Each prompt must contain a string 'prompt' field."
                    )

                prompts.append(prompt)

            return prompts

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompts file not found: {path}"
        )
