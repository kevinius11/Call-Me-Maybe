from src.llm import LLM
from src.decoding import Decoder, DecoderState, DecodingState
from src.schemas import FunctionDefinition
from src.prompt_builder import build_prompt


class GenerationError(Exception):
    """Representa un error durante la generación."""
    pass


class Generator:
    """Orquesta la generacion restringida del JSON."""
    def __init__(self,
                 llm: LLM,
                 decoder: Decoder,
                 functions: list[FunctionDefinition]
                 ) -> None:
        """
        Inicializa el generador.

        Args:
            llm: Modelo utilizado para generar tokens.
            decoder: Decoder encargado de restringir la generación.
            functions: Funciones disponibles para el modelo.
        """
        self._llm = llm
        self._decoder = decoder
        self._functions = functions

    def _encode_fixed_text(
            self,
            text: str,
    ) -> list[int]:
        """
        Tokeniza un fragmento fijo del formato de generación.

        Args:
            text: Texto fijo que debe formar parte del contexto.

        Returns:
            Lista de token IDs correspondiente al texto.
        """
        return self._llm.encode(text)

    def generate(
            self,
            prompt: str,
    ) -> str:
        """Genera una llamada JSON restringida."""
        semantic_prompt = build_prompt(
            prompt,
            self._functions,
        )

        input_ids = self._llm.encode(
            semantic_prompt
        )

        fixed_prefix = '{"fn_name": "'

        input_ids.extend(
            self._encode_fixed_text(fixed_prefix)
        )

        state = DecoderState(
            phase=DecodingState.EXPECT_FN_NAME
        )
