import json


class Vocabulary:
    """Proporciona acceso al vocabulario del modelo."""
    def __init__(self, vocabulary_path: str):
        """
        Inicializa el vocabulario a partir de un archivo JSON.

        Args:
            vocabulary_path: Ruta al archivo JSON que contiene el vocabulario.

        Raises:
            ValueError: Si no se puede cargar o procesar el vocabulario.
        """
        try:
            with open(vocabulary_path, "r", encoding="utf-8") as file:
                token_to_id = json.load(file)

            self._token_to_id = token_to_id
            self._id_to_token = {
                token_id: token
                for token, token_id in token_to_id.items()
            }

        except Exception as e:
            raise ValueError(
                f"Error al cargar vocabulario: {e}"
            ) from e

    def get_token(self, token_id: int) -> str:
        """
        Obtiene el token asociado a un identificador.

        Args:
            token_id: Identificador numérico del token.

        Returns:
            El texto correspondiente al token.

        Raises:
            KeyError: Si el identificador no existe en el vocabulario.
        """
        try:
            return self._id_to_token[token_id]
        except KeyError as e:
            raise KeyError(f"Token ID no encontrado: {token_id}") from e

    def get_token_id(self, token: str) -> int:
        """
        Obtiene el identificador asociado a un token.

        Args:
            token: Texto del token.

        Returns:
            El identificador numérico correspondiente al token.

        Raises:
            KeyError: Si el token no existe en el vocabulario.
        """
        try:
            return self._token_to_id[token]
        except KeyError as e:
            raise KeyError(f"Token no encontrado: {token}") from e

    def __len__(self) -> int:
        """
        Obtiene el número total de tokens del vocabulario.

        Returns:
            Número de tokens disponibles.
        """
        return len(self._token_to_id)
