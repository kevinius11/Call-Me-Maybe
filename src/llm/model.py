from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np


class LLMError(Exception):
    pass


class LLM:
    """
    Encapsula el SDK del modelo y expone las operaciones
    de inferencia necesarias para el resto del sistema.
    """
    def __init__(self) -> None:
        """
        Inicializa el wrapper del LLM y carga el modelo.

        Raises:
            LLMError: Si ocurre un error durante la inicializacion
            del modelo.
        """
        try:
            self._model = Small_LLM_Model()
        except Exception as e:
            raise LLMError(f"Error al instanciar el LLM: {e}") from e

    def encode(self, text: str) -> list[int]:
        """
        Tokeniza  un texto y devuelve sus token IDs.

        Args:
            text: Texto que se desea tokenizar.
        Returns:
            Lista de enteros donde cada entero representa un token ID.
        Raises:
            LLMError: Si ocurre un error durante la tokenizacion.
        """
        try:
            input_ids = self._model.encode(text)
            return input_ids.squeeze(0).tolist()
        except Exception as e:
            raise LLMError(f"Error encoding text: {e}") from e

    def decode(self, token_ids: list[int]) -> str:
        """
        Decodifica una lista de token IDs y devuelve
        el texto correspondiente

        Args:
            token_ids: Lista de enteros que representan tokens

        Returns:
            Texto reconstruido a partir de los token IDs.

        Raises:
            LLMError: Si ocurre un error durante la decodificacion.
        """
        try:
            return self._model.decode(token_ids)
        except Exception as e:
            raise LLMError(f"Error decoding token IDs: {e}") from e

    def get_logits(self, input_ids: list[int]) -> np.ndarray:
        """
        
        """