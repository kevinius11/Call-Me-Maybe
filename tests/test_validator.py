from src.schemas import FunctionDefinition, ParameterDefinition
from src.validation import SemanticValidator, ValidationError


def build_validator() -> SemanticValidator:
    function = FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers",
        parameters={
            "a": ParameterDefinition(type="number"),
            "b": ParameterDefinition(type="number"),
        },
        returns=ParameterDefinition(type="number"),
    )

    return SemanticValidator([function])


def test_valid_call() -> None:
    validator = build_validator()

    validator.validate({
        "fn_name": "fn_add_numbers",
        "args": {
            "a": 2,
            "b": 3,
        },
    })

    print("1. valid call: OK")


def test_invalid_function() -> None:
    validator = build_validator()

    try:
        validator.validate({
            "fn_name": "fn_unknown",
            "args": {
                "a": 2,
                "b": 3,
            },
        })
    except ValidationError:
        print("2. invalid function: OK")
        return

    raise AssertionError("No se detectó una función inexistente.")


def test_missing_parameter() -> None:
    validator = build_validator()

    try:
        validator.validate({
            "fn_name": "fn_add_numbers",
            "args": {
                "a": 2,
            },
        })
    except ValidationError:
        print("3. missing parameter: OK")
        return

    raise AssertionError("No se detectó un parámetro ausente.")


def test_extra_parameter() -> None:
    validator = build_validator()

    try:
        validator.validate({
            "fn_name": "fn_add_numbers",
            "args": {
                "a": 2,
                "b": 3,
                "c": 4,
            },
        })
    except ValidationError:
        print("4. extra parameter: OK")
        return

    raise AssertionError("No se detectó un parámetro extra.")


def test_invalid_type() -> None:
    validator = build_validator()

    try:
        validator.validate({
            "fn_name": "fn_add_numbers",
            "args": {
                "a": "hello",
                "b": 3,
            },
        })
    except ValidationError:
        print("5. invalid type: OK")
        return

    raise AssertionError("No se detectó un tipo incorrecto.")


def test_extra_top_level_key() -> None:
    validator = build_validator()

    try:
        validator.validate({
            "fn_name": "fn_add_numbers",
            "args": {
                "a": 2,
                "b": 3,
            },
            "extra": "not allowed",
        })
    except ValidationError:
        print("6. extra top-level key: OK")
        return

    raise AssertionError("No se detectó una clave adicional.")


if __name__ == "__main__":
    test_valid_call()
    test_invalid_function()
    test_missing_parameter()
    test_extra_parameter()
    test_invalid_type()
    test_extra_top_level_key()
    print("Todas las pruebas de validation han pasado.")
