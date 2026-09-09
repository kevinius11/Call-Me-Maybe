import json

import numpy as np

from src.llm import LLM
from src.prompt_builder import build_prompt
from src.schemas import FunctionDefinition


def load_functions() -> list[FunctionDefinition]:
    with open("functions_definition.json", "r") as file:
        data = json.load(file)

    return [FunctionDefinition(**function) for function in data]


def generate_unconstrained(
    llm: LLM,
    prompt: str,
    max_tokens: int = 80,
) -> str:
    input_ids = llm.encode(prompt)
    original_length = len(input_ids)

    for _ in range(max_tokens):
        logits = llm.get_logits(input_ids)
        next_token_id = int(np.argmax(logits))
        input_ids.append(next_token_id)

    return llm.decode(input_ids[original_length:])


def main() -> None:
    functions = load_functions()
    llm = LLM()

    tests = [
        "Replace all vowels in 'Programming is fun' with asterisks",
        "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
    ]

    for test in tests:
        print("=" * 80)
        print(f"USER: {test}")
        print("=" * 80)

        prompt = build_prompt(test, functions)

        print("\nMODEL OUTPUT (unconstrained):\n")
        print(generate_unconstrained(llm, prompt))
        print()


if __name__ == "__main__":
    main()
