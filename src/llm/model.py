from llm_sdk.llm_sdk import Small_LLM_Model


class LLMError(Exception):
    pass


class LLM:
    """
    Encapsula el SDK del modelo y expone las operaciones
    de inferencia necesarias para el resto del sistema.
    """
    def __init__(self, ):
        try:
            self._model = Small_LLM_Model()
        except Exception as e:
            raise LLMError(f"Error al instanciar el LLM: {e}")