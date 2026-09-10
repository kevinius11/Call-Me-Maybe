from src.schemas import FunctionDefinition


class ValidationError(Exception):
    """Se produce cuando una llamada de función no es válida."""
    pass


class SemanticValidator:
    """Valida llamadas de función según sus definiciones."""

    def __init__(
            self,
            functions: list[FunctionDefinition],
    ) -> None:
        """
        Inicializa el validador con las funciones disponibles.

        Args:
            functions: Definiciones de las funciones disponibles.
        """
        self._functions = {
            function.name: function
            for function in functions
        }

    def validate(self, data: object) -> None:
        """
        Valida una llamada de funcion:

        Args:
            data: JSON decodificado que representa la llamada.

        Raises:
            ValidationError: Si la llamada no cumple el schema semantico.
        """
        if not isinstance(data, dict):
            raise ValidationError(
                "La llamada generada debe ser un objeto JSON."
            )

        if set(data.keys()) != {"fn_name", "args"}:
            raise ValidationError(
                "La llamada debe contener exactamente "
                "'fn_name' y 'args'."
            )

        fn_name = data.get("fn_name")
        args = data.get("args")

        if not isinstance(fn_name, str):
            raise ValidationError(
                "'fn_namee' debe ser un string."
            )

        function = self._functions.get(fn_name)

        if function is None:
            raise ValidationError(
                f"Funcion no encontrada: {fn_name}"
            )

        if not isinstance(args, dict):
            raise ValidationError(
                "'args' debe ser un objeto JSON."
            )

        expected_parameters = function.parameters
        actual_parameters = set(args.keys())

        if actual_parameters != set(expected_parameters.keys()):
            missing = set(expected_parameters.keys()) - actual_parameters
            extra = actual_parameters - set(expected_parameters.keys())

            if missing:
                raise ValidationError(
                    f"Faltan parametros requeridos: {sorted(missing)}"
                )

            if extra:
                raise ValidationError(
                    f"Parametros no encontrados: {sorted(extra)}"
                )

        for parameter_name, parameter in expected_parameters.items():
            value = args[parameter_name]

            if not self._valid_type(value, parameter.type):
                raise ValidationError(
                    f"Tipo invalido para '{parameter_name}': "
                    f"se esperaba {parameter.type}"
                )

    def _valid_type(
        self,
        value: object,
        expected_type: str,
    ) -> bool:
        """
        Comprueba si un valor coincide con el tipo esperado.

        Args:
            value: Valor del parámetro que se va a comprobar.
            expected_type: Tipo esperado según la definición.

        Returns:
            True si el tipo es válido y False en caso contrario.

        Raises:
            ValidationError: Si el tipo esperado no está soportado.
        """
        if expected_type == "number":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        if expected_type == "string":
            return isinstance(value, str)

        if expected_type == "boolean":
            return isinstance(value, bool)

        raise ValidationError(
            f"Tipo de parametro no soportado: {expected_type}"
        )
