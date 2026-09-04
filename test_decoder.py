import json

from src.decoding import Decoder, DecoderState, DecodingState
from src.llm import LLM, Vocabulary
from src.schemas import FunctionDefinition


def load_functions() -> list[FunctionDefinition]:
    """
    Carga las definiciones de funciones desde el archivo JSON.

    Returns:
        Lista de definiciones de funciones.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de definiciones.
        json.JSONDecodeError: Si el archivo contiene JSON inválido.
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


def test_number(
    decoder: Decoder,
    vocabulary: Vocabulary,
) -> None:
    """
    Comprueba la finalización y transición de un valor numérico.

    Args:
        decoder: Decoder utilizado para validar y actualizar estados.
        vocabulary: Vocabulario utilizado para obtener token IDs.

    Returns:
        None.

    Raises:
        DecoderError: Si la transición del decoder no es válida.
    """
    state = DecoderState(
        phase=DecodingState.EXPECT_ARGS_VALUE,
        selected_function="fn_add_numbers",
        current_parameter="a",
        prefix="42",
    )

    print("NUMBER")

    print(
        "can_finish_value:",
        decoder.can_finish_value(state),
    )

    print(
        "comma valid:",
        decoder._token_is_valid(",", state),
    )

    comma_id = vocabulary.get_token_id(",")

    new_state = decoder.update_state(
        comma_id,
        state,
    )

    print(
        "after comma:",
        new_state.phase,
    )

    print(
        "prefix:",
        repr(new_state.prefix),
    )

    print("-" * 50)


def test_string(
    decoder: Decoder,
    vocabulary: Vocabulary,
) -> None:
    """
    Comprueba el cierre y transición de un valor string.

    Args:
        decoder: Decoder utilizado para validar y actualizar estados.
        vocabulary: Vocabulario utilizado para obtener token IDs.

    Returns:
        None.

    Raises:
        DecoderError: Si la transición del decoder no es válida.
    """
    state = DecoderState(
        phase=DecodingState.EXPECT_ARGS_VALUE,
        selected_function="fn_greet",
        current_parameter="name",
        prefix="Shrek",
    )

    print("STRING")

    print(
        "quote valid:",
        decoder._token_is_valid('"', state),
    )

    quote_id = vocabulary.get_token_id('"')

    new_state = decoder.update_state(
        quote_id,
        state,
    )

    print(
        "after quote:",
        new_state.phase,
    )

    print(
        "prefix:",
        repr(new_state.prefix),
    )

    print("-" * 50)


def test_number_values(
    decoder: Decoder,
) -> None:
    """
    Comprueba distintos estados de construcción de números.

    Args:
        decoder: Decoder utilizado para realizar las validaciones.

    Returns:
        None.

    Raises:
        DecoderError: Si el estado utilizado no es válido.
    """
    prefixes = [
        "",
        "-",
        "4",
        "42",
        "42.",
        "42.5",
    ]

    print("NUMBER VALUES")

    for prefix in prefixes:
        state = DecoderState(
            phase=DecodingState.EXPECT_ARGS_VALUE,
            selected_function="fn_add_numbers",
            current_parameter="a",
            prefix=prefix,
        )

        print(
            "prefix:",
            repr(prefix),
            "can_finish:",
            decoder.can_finish_value(state),
            "comma:",
            decoder._token_is_valid(",", state),
            "brace:",
            decoder._token_is_valid("}", state),
        )

    print("-" * 50)


def test_last_number(
    decoder: Decoder,
) -> None:
    """
    Comprueba los delimitadores permitidos para el último número.

    Args:
        decoder: Decoder utilizado para realizar las validaciones.

    Returns:
        None.

    Raises:
        DecoderError: Si el estado utilizado no es válido.
    """
    state = DecoderState(
        phase=DecodingState.EXPECT_ARGS_VALUE,
        selected_function="fn_add_numbers",
        current_parameter="b",
        prefix="42",
    )

    print("LAST NUMBER")

    print(
        "can_finish:",
        decoder.can_finish_value(state),
    )

    print(
        "comma:",
        decoder._token_is_valid(",", state),
    )

    print(
        "brace:",
        decoder._token_is_valid("}", state),
    )

    print("-" * 50)


def test_full_state_sequence(
    decoder: Decoder,
    vocabulary: Vocabulary,
    llm: LLM,
) -> None:
    """
    Comprueba una secuencia completa de estados del decoder.

    Args:
        decoder: Decoder utilizado para actualizar los estados.
        vocabulary: Vocabulario utilizado para interpretar los tokens.
        llm: Modelo utilizado para tokenizar los fragmentos de texto.

    Returns:
        None.

    Raises:
        DecoderError: Si un token no puede procesarse en el estado actual.
    """
    state = DecoderState(
        phase=DecodingState.EXPECT_FN_NAME,
    )

    sequences = [
        ("fn_add_numbers", DecodingState.EXPECT_FN_NAME),
        ('"', DecodingState.EXPECT_ARGS_KEY),
        ("a", DecodingState.EXPECT_ARGS_KEY),
        ('"', DecodingState.EXPECT_ARGS_VALUE),
        ("40", DecodingState.EXPECT_ARGS_VALUE),
        (",", DecodingState.EXPECT_ARGS_KEY),
        ("b", DecodingState.EXPECT_ARGS_KEY),
        ('"', DecodingState.EXPECT_ARGS_VALUE),
        ("2", DecodingState.EXPECT_ARGS_VALUE),
        ("}", DecodingState.DONE),
    ]

    print("FULL STATE SEQUENCE")

    for text, expected_phase in sequences:
        token_ids = llm.encode(text)

        for token_id in token_ids:
            token_string = vocabulary.get_token(token_id)

            state = decoder.update_state(
                token_id,
                state,
            )

            print(
                "TOKEN:",
                repr(token_string),
                "STATE:",
                state.phase,
            )

        print(
            "EXPECTED:",
            expected_phase,
            "PASS:",
            state.phase == expected_phase,
        )

        print("-" * 40)


def main() -> None:
    """
    Ejecuta todas las pruebas del decoder.

    Returns:
        None.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de funciones.
        json.JSONDecodeError: Si las definiciones contienen JSON inválido.
        DecoderError: Si alguna prueba produce una transición inválida.
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

    test_number(
        decoder,
        vocabulary,
    )

    test_string(
        decoder,
        vocabulary,
    )

    test_number_values(
        decoder,
    )

    test_last_number(
        decoder,
    )

    test_full_state_sequence(
        decoder,
        vocabulary,
        llm,
    )


if __name__ == "__main__":
    main()
