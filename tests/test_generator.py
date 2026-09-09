import json

from src.decoding import Decoder
from src.generation import Generator
from src.llm import LLM, Vocabulary
from src.schemas import FunctionDefinition


def load_functions() -> list[FunctionDefinition]:
    """
    Carga las definiciones de funciones desde el archivo JSON.

    Returns:
        Lista de definiciones de funciones.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo.
        json.JSONDecodeError: Si el JSON no es válido.
    """
    with open(
        "functions_definition.json",
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return [
        FunctionDefinition(**item)
        for item in data
    ]


def main() -> None:
    """
    Ejecuta todos los casos de prueba de generación.

    Returns:
        None.

    Raises:
        GenerationError: Si una generación no puede completarse.
        DecoderError: Si ocurre un error durante la decodificación.
    """
    llm = LLM()

    vocabulary = Vocabulary(
        llm.get_vocab_path()
    )

    functions = load_functions()

    decoder = Decoder(
        functions=functions,
        vocabulary=vocabulary,
    )

    generator = Generator(
        llm=llm,
        decoder=decoder,
        functions=functions,
        vocabulary=vocabulary,
    )

    with open(
        "function_calling_tests.json",
        "r",
        encoding="utf-8",
    ) as file:
        tests = json.load(file)

    for index, test in enumerate(tests, start=1):
        prompt = test["prompt"]

        print()
        print("=" * 70)
        print(f"TEST {index}")
        print("=" * 70)
        print(f"PROMPT: {prompt}")

        result = generator.generate(prompt)

        print(f"RESULT: {result}")


if __name__ == "__main__":
    main()
