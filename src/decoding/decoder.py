import numpy as np

from src.llm import Vocabulary
from src.schemas import FunctionDefinition

from .state import DecoderState


def apply_constraints(
    logits: np.ndarray,
    state: DecoderState,
    functions: list[FunctionDefinition],
    vocabulary: Vocabulary,
) -> np.ndarray:
    """
    Aplica las restricciones de decodificación a los logits.

    Args:
        logits: Array de NumPy con los logits del modelo.
        state: Estado actual de la máquina de decodificación.
        functions: Definiciones de funciones disponibles.
        vocabulary: Vocabulario utilizado por el modelo.

    Returns:
        Array de NumPy con los logits modificados según las restricciones.
    """
    pass


def update_state(
    token_id: int,
    state: DecoderState,
    functions: list[FunctionDefinition],
    vocabulary: Vocabulary,
) -> DecoderState:
    """
    Actualiza el estado de la máquina de decodificación tras seleccionar
    un token.

    Args:
        token_id: Identificador del token seleccionado.
        state: Estado actual de la máquina de decodificación.
        functions: Definiciones de funciones disponibles.
        vocabulary: Vocabulario utilizado por el modelo.

    Returns:
        Nuevo estado de la máquina de decodificación.
    """
    pass
