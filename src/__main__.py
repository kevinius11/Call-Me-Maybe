import argparse
import json
from pathlib import Path

from src.decoding import Decoder
from src.generation import Generator
from src.input import load_function_definitions, load_prompts
from src.llm import LLM, Vocabulary
from src.schemas import FunctionCallResult, FunctionDefinition


DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_FUNCTIONS = "data/input/function_definitions.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate structured function calls from prompts."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def build_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions."""
    raw_functions = load_function_definitions(path)

    return [
        FunctionDefinition(**function)
        for function in raw_functions
    ]


def build_generator(
    functions: list[FunctionDefinition],
) -> Generator:
    """Build the function call generator."""
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
) -> list[FunctionCallResult]:
    """Generate and validate results for all prompts."""
    results: list[FunctionCallResult] = []

    for prompt in prompts:
        generated = generator.generate(prompt)
        data = json.loads(generated)

        result = FunctionCallResult(
            prompt=prompt,
            fn_name=data["fn_name"],
            args=data["args"],
        )
        results.append(result)

    return results


def save_results(
    results: list[FunctionCallResult],
    path: str,
) -> None:
    """Save generated results as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        result.model_dump()
        for result in results
    ]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    """Run the complete function-calling pipeline."""
    args = parse_arguments()

    prompts = load_prompts(args.input)
    functions = build_functions(DEFAULT_FUNCTIONS)
    generator = build_generator(functions)

    results = process_prompts(prompts, generator)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
