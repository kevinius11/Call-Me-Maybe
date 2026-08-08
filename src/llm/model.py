from llm_sdk.llm_sdk import Small_LLM_Model


class LLMError(Exception):
    pass


class LLM:
    """
    Encapsula el SDK del modelo y expone las operaciones
    de inferencia necesarias para el resto del sistema.
    """
    def __init__(self, ):
        """

        """
        try:
            self._model = Small_LLM_Model()
        except Exception as e:
            raise LLMError(f"Error al instanciar el LLM: {e}")

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
