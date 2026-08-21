import numpy as np

from src.llm import Vocabulary
from src.schemas import FunctionDefinition

from .state import DecoderState, DecodingState


class DecoderError(Exception):
    """Representa un error durante la decodificacion restringida"""
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
                    "phase": DecodingState.EXPECT_SEPARATOR
                }
            )

        return state

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

        raise DecoderError(f"Estado no soportado: {state.phase}")
