import numpy as np

from src.llm import Vocabulary
from src.schemas import FunctionDefinition

from .state import DecoderState, DecodingState


class DecoderError(Exception):
    """Representa un error durante la decodificación restringida."""
    pass


class Decoder:
    """Gestiona la decodificación restringida del modelo."""

    def __init__(
        self,
        functions: list[FunctionDefinition],
        vocabulary: Vocabulary,
    ) -> None:
        """
        Inicializa el decoder con las funciones y el vocabulario disponibles.

        Args:
            functions: Definiciones de las funciones que pueden ser
                seleccionadas durante la generación.
            vocabulary: Vocabulario utilizado por el modelo.
        """
        self._functions = functions
        self._vocabulary = vocabulary
        self._names = [function.name for function in functions]
        self._names_complete = {
            function.name for function in functions
        }
        self._functions_by_name = {
            function.name: function
            for function in functions
        }

    def apply_constraints(
        self,
        logits: np.ndarray,
        state: DecoderState,
    ) -> np.ndarray:
        """
        Aplica las restricciones de decodificación a los logits actuales.

        Args:
            logits: Array de NumPy con los logits del modelo.
            state: Estado actual de la máquina de decodificación.

        Returns:
            Array de NumPy con los logits modificados según las
            restricciones del estado actual.
        """
        constrained_logits = logits.copy()

        for token_id in range(len(self._vocabulary)):
            token_string = self._vocabulary.get_token(token_id)

            if not self._token_is_valid(token_string, state):
                constrained_logits[token_id] = -np.inf

        return constrained_logits

    def update_state(
        self,
        token_id: int,
        state: DecoderState,
    ) -> DecoderState:
        """
        Actualiza el estado de la máquina tras la selección de un token.

        Args:
            token_id: Identificador del token seleccionado.
            state: Estado actual de la máquina de decodificación.

        Returns:
            Nuevo estado de la máquina de decodificación.
        """
        token_string = self._vocabulary.get_token(token_id)

        if state.phase == DecodingState.EXPECT_FN_NAME:
            if token_string != '"':
                return state.model_copy(
                    update={"prefix": state.prefix + token_string}
                )

            return state.model_copy(
                update={
                    "selected_function": state.prefix,
                    "prefix": "",
                    "phase": DecodingState.EXPECT_SEPARATOR,
                }
            )

        if state.phase == DecodingState.EXPECT_ARGS_KEY:
            if token_string != '"':
                return state.model_copy(
                    update={"prefix": state.prefix + token_string}
                )

            return state.model_copy(
                update={
                    "current_parameter": state.prefix,
                    "prefix": "",
                    "phase": DecodingState.EXPECT_SEPARATOR,
                }
            )

        if state.phase == DecodingState.EXPECT_ARGS_VALUE:
            if state.current_parameter is None:
                raise DecoderError(
                    "No hay un parámetro seleccionado en EXPECT_ARGS_VALUE"
                )

            if state.selected_function is None:
                raise DecoderError(
                    "No hay una función seleccionada en EXPECT_ARGS_VALUE"
                )

            function = self._functions_by_name.get(
                state.selected_function
            )

            if function is None:
                raise DecoderError(
                    f"Función no encontrada: {state.selected_function}"
                )

            parameter = function.parameters.get(
                state.current_parameter
            )

            if parameter is None:
                raise DecoderError(
                    f"Parámetro no encontrado: {state.current_parameter}"
                )

            parameter_type = parameter.type

            if parameter_type == "number":
                if token_string not in {",", "}"}:
                    return state.model_copy(
                        update={"prefix": state.prefix + token_string}
                    )

                parameter_names = list(function.parameters.keys())
                current_index = parameter_names.index(
                    state.current_parameter
                )
                has_more_parameters = (
                    current_index < len(parameter_names) - 1
                )

                if token_string == ",":
                    if not has_more_parameters:
                        raise DecoderError(
                            "Separador ',' sin parámetros restantes"
                        )

                    return state.model_copy(
                        update={
                            "prefix": "",
                            "phase": DecodingState.EXPECT_ARGS_KEY,
                        }
                    )

                if token_string == "}":
                    if has_more_parameters:
                        raise DecoderError(
                            "Cierre de argumentos antes de completar "
                            "todos los parámetros"
                        )

                    return state.model_copy(
                        update={
                            "prefix": "",
                            "phase": DecodingState.DONE,
                        }
                    )

                if parameter_type == "string":
                    if token_string != '"':
                        return state.model_copy(
                            update={"prefix": state.prefix + token_string}
                        )

                    return state.model_copy(
                        update={
                            "prefix": "",
                            "phase": DecodingState.EXPECT_SEPARATOR,
                        }
                    )

                if parameter_type == "boolean":
                    raise DecoderError(
                        "La actualización de valores booleanos aún no está implementada"
                    )

                raise DecoderError(
                    f"Tipo de parámetro no soportado: {parameter_type}"
                )

        raise DecoderError(
            f"Estado no soportado: {state.phase}"
        )

    def _token_is_valid(
        self,
        token_string: str,
        state: DecoderState,
    ) -> bool:
        """
        Determina si un token puede generarse en el estado actual.

        Args:
            token_string: Representación textual del token.
            state: Estado actual de la máquina de decodificación.

        Returns:
            True si el token es válido para la generación actual;
            False en caso contrario.
        """

        # 1. EXPECT_FN_NAME
        if state.phase == DecodingState.EXPECT_FN_NAME:
            candidate = state.prefix + token_string

            continues_name = any(
                name.startswith(candidate)
                for name in self._names
            )

            closes_name = (
                state.prefix in self._names_complete
                and token_string == '"'
            )

            return continues_name or closes_name
        
        # 2. EXPECT_ARGS_KEY
        if state.phase == DecodingState.EXPECT_ARGS_KEY:
            if state.selected_function is None:
                raise DecoderError(
                    "No hay una función seleccionada en EXPECT_ARGS_KEY"
                )

            function = self._functions_by_name.get(
                state.selected_function
            )

            if function is None:
                raise DecoderError(
                    f"Función no encontrada: {state.selected_function}"
                )

            valid_keys = function.parameters.keys()
            candidate = state.prefix + token_string

            continues_key = any(
                key.startswith(candidate)
                for key in valid_keys
            )

            closes_key = (
                state.prefix in function.parameters
                and token_string == '"'
            )

            return continues_key or closes_key

        # 3. EXPECT_ARGS_VALUE
        if state.phase == DecodingState.EXPECT_ARGS_VALUE:
            if state.current_parameter is None:
                raise DecoderError(
                    "No hay un parámetro seleccionado en EXPECT_ARGS_VALUE"
                )

            if state.selected_function is None:
                raise DecoderError(
                    "No hay una función seleccionada en EXPECT_ARGS_VALUE"
                )

            function = self._functions_by_name.get(
                state.selected_function
            )

            if function is None:
                raise DecoderError(
                    f"Función no encontrada: {state.selected_function}"
                )

            parameter = function.parameters.get(
                state.current_parameter
            )

            if parameter is None:
                raise DecoderError(
                    f"Parámetro no encontrado: {state.current_parameter}"
                )

            parameter_type = parameter.type

            if parameter_type == "number":
                return self._number_token_is_valid(
                    token_string,
                    state.prefix
                )

            if parameter_type == "string":
                return self._string_token_is_valid(token_string)

            if parameter_type == "boolean":
                raise DecoderError(
                    "Aun no esta implementado"
                )

            raise DecoderError(
                f"Tipo de parámetro no soportado: {parameter_type}"
            )

        # 4. Ningún estado reconocido
        raise DecoderError(
            f"Estado no soportado: {state.phase}"
        )

    def _number_token_is_valid(
        self,
        token_string: str,
        prefix: str,
    ) -> bool:
        """
        Determina si un token puede continuar la construcción de un número.

        Args:
            token_string: Representación textual del token candidato.
            prefix: Parte del número generada hasta el momento.

        Returns:
            True si el token es válido para continuar o cerrar el número;
            False en caso contrario.
        """
        if prefix == "":
            return token_string.isdigit() or token_string == "-"

        if prefix == "-":
            return token_string.isdigit()

        if prefix.endswith("."):
            return token_string.isdigit()

        if token_string in {",", "}"}:
            return True

        if token_string.isdigit():
            return True

        if token_string == ".":
            return "." not in prefix

        return False

    def _string_token_is_valid(
            self,
            token_string: str,
    ) -> bool:
        """
        Determina si un token es válido dentro de un string JSON.

        Args:
            token_string: Representación textual del token candidato.

        Returns:
            True si el token puede aparecer dentro del string;
            False si el token cierra el string.
        """
        return token_string != '"'
