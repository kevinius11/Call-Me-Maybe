from src.llm import LLM, Vocabulary
from src.decoding import Decoder, DecoderState, DecodingState
from src.schemas import FunctionDefinition
from src.generation import Generator


def main() -> None:
    llm = LLM()

    # De momento no necesitamos funciones ni decoder
    # para probar únicamente _sync_states().
    generator = Generator(
        llm=llm,
        decoder=None,
        functions=[],
        vocabulary=None,
    )

    tests = [
        (
            "FN_NAME -> ARGS_KEY",
            DecoderState(
                phase=DecodingState.EXPECT_FN_NAME
            ),
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_KEY
            ),
            '"',
        ),
        (
            "ARGS_KEY -> ARGS_VALUE",
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_KEY
            ),
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_VALUE
            ),
            '"',
        ),
        (
            "NUMBER + comma",
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_VALUE
            ),
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_KEY
            ),
            ",",
        ),
        (
            "STRING + quote",
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_VALUE
            ),
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_KEY
            ),
            '"',
        ),
        (
            "NUMBER + closing brace",
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_VALUE
            ),
            DecoderState(
                phase=DecodingState.DONE
            ),
            "}",
        ),
        (
            "STRING + closing quote",
            DecoderState(
                phase=DecodingState.EXPECT_ARGS_VALUE
            ),
            DecoderState(
                phase=DecodingState.DONE
            ),
            '"',
        ),
    ]

    for name, previous_state, state, token_string in tests:
        input_ids = []

        generator._sync_states(
            input_ids,
            previous_state,
            state,
            token_string,
        )

        result = llm.decode(input_ids)

        print(name)
        print("TOKEN :", repr(token_string))
        print("RESULT:", repr(result))
        print("-" * 40)


if __name__ == "__main__":
    main()