import argparse
import json

from src.decoding import Decoder
from src.generation import Generator
from src.input import load_function_definitions, load_prompts
from src.llm import LLMError, LLM, Vocabulary
from src.output import OutputError, save_results
from src.schemas import FunctionCallResult, FunctionDefinition
from src.validation import ValidationError, SemanticValidator


DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_FUNCTIONS = "data/input/function_definitions.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_arguments() -> argparse.Namespace:
    """
    Procesa los argumentos recibidos desde la línea de comandos.

    Returns:
        Argumentos procesados.
    """
    parser = argparse.ArgumentParser(
        description="Genera llamadas de función estructuradas "
                    "a partir de prompts.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Ruta al archivo JSON de prompts.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Ruta al archivo JSON de salida.",
    )
    return parser.parse_args()


def build_functions(
    path: str,
) -> list[FunctionDefinition]:
    """
    Carga y valida las definiciones de funciones.

    Args:
        path: Ruta al archivo JSON de definiciones.

    Returns:
        Lista de definiciones de funciones validadas.
    """
    raw_functions = load_function_definitions(path)

    return [
        FunctionDefinition(**function)
        for function in raw_functions
    ]


def build_generator(
    functions: list[FunctionDefinition],
) -> Generator:
    """
    Construye el generador de llamadas de función.

    Args:
        functions: Definiciones de funciones disponibles.

    Returns:
        Generador configurado.
    """
    llm = LLM()
    vocabulary = Vocabulary(llm.get_vocab_path())
    decoder = Decoder(functions, vocabulary)

    return Generator(
        llm=llm,
        decoder=decoder,
        functions=functions,
        vocabulary=vocabulary,
    )


def process_prompts(
    prompts: list[str],
    generator: Generator,
    validator: SemanticValidator,
) -> list[FunctionCallResult]:
    """
    Genera y valida una llamada de función para cada prompt.

    Args:
        prompts: Lista de solicitudes en lenguaje natural.
        generator: Generador de llamadas restringidas.
        validator: Validador semántico de las llamadas.

    Returns:
        Lista de resultados de llamadas validadas.

    Raises:
        ValueError: Si la salida generada no contiene un JSON válido.
    """
    results: list[FunctionCallResult] = []

    for prompt in prompts:
        generated = generator.generate(prompt)

        try:
            data = json.loads(generated)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "La generación no produjo un JSON válido "
                f"para el prompt: {prompt}"
            ) from exc

        validator.validate(data)

        result = FunctionCallResult(
            prompt=prompt,
            fn_name=data["fn_name"],
            args=data["args"],
        )

        results.append(result)

    return results


def main() -> None:
    """Ejecuta el flujo completo de generación de llamadas."""
    try:
        args = parse_arguments()

        prompts = load_prompts(args.input)
        functions = build_functions(DEFAULT_FUNCTIONS)

        generator = build_generator(functions)
        validator = SemanticValidator(functions)

        results = process_prompts(
            prompts,
            generator,
            validator,
        )

        save_results(
            results,
            args.output,
        )

    except (
        FileNotFoundError,
        ValueError,
        ValidationError,
        OutputError,
        LLMError,
    ) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
