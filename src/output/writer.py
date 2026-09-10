import json
from pathlib import Path

from src.schemas import FunctionCallResult


class OutputError(Exception):
    """Se produce cuando no se puede escribir el archivo de salida."""
    pass


def save_results(
    results: list[FunctionCallResult],
    path: str,
) -> None:
    """Guarda los resultados validados en un archivo JSON.

    Args:
        results: Lista de resultados validados.
        path: Ruta del archivo JSON de salida.

    Raises:
        OutputError: Si no se puede escribir el archivo.
    """
    output_path = Path(path)

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                [result.model_dump() for result in results],
                file,
                indent=2,
                ensure_ascii=False,
            )

    except OSError as exc:
        raise OutputError(
            f"No se pudo escribir el archivo de salida: {path}"
        ) from exc
