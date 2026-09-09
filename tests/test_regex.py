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
    Ejecuta una prueba aislada del caso de sustitución.

    Returns:
        None.
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

    prompt = (
        'Replace all numbers in '
        '"Hello 34 I\'m 233 years old" '
        'with NUMBERS'
    )

    result = generator.generate(prompt)

    print("FINAL RESULT:")
    print(result)


if __name__ == "__main__":
    main()
