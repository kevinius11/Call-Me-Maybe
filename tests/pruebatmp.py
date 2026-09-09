from src.llm import LLM, Vocabulary
from src.prompt_builder import build_prompt
from src.schemas import FunctionDefinition
from src.decoding import Decoder, DecoderState, DecodingState

import json
import numpy as np


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
    Ejecuta una prueba de diagnóstico sobre la selección de función.

    Returns:
        None.
    """
    llm = LLM()

    vocabulary = Vocabulary(
        llm.get_vocab_path()
    )

    functions = load_functions()

    semantic_prompt = build_prompt(
        "Add 2 and 3?",
        functions,
    )

    decoder = Decoder(
        functions=functions,
        vocabulary=vocabulary,
    )

    input_ids = llm.encode(
        semantic_prompt
    )

    input_ids.extend(
        llm.encode(
            '{"fn_name": "'
        )
    )

    fn_token_id = vocabulary.get_token_id(
        "fn"
    )

    input_ids.append(
        fn_token_id
    )

    state = DecoderState(
        phase=DecodingState.EXPECT_FN_NAME,
        prefix="fn",
    )

    print("INPUT IDS:", input_ids)

    logits = llm.get_logits(
        input_ids
    )

    constrained_logits = decoder.apply_constraints(
        logits,
        state,
    )

    valid_ids = np.flatnonzero(
        np.isfinite(constrained_logits)
    )

    top_ids = valid_ids[
        np.argsort(
            constrained_logits[valid_ids]
        )[::-1]
    ][:20]

    for token_id in top_ids:
        token = vocabulary.get_token(
            int(token_id)
        )

        print(
            int(token_id),
            repr(token),
            float(constrained_logits[token_id]),
        )


if __name__ == "__main__":
    main()
