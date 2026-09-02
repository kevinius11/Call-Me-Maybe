from src.llm import LLM
from src.decoding import Decoder, DecoderState, DecodingState
from src.schemas import FunctionDefinition
from src.prompt_builder import build_prompt

import numpy as np


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

    def _sync_states(
            self,
            input_ids: list[int],
            previous_state: DecoderState,
            state: DecoderState,
            token_id: int,
    ) -> None:
        """
        Sincroniza la generación con la transición de estados actual.

        Args:
            input_ids: Lista de token IDs que representa el contexto actual
                de generación.
            previous_state: Estado de la máquina antes de procesar el token.
            state: Estado de la máquina después de procesar el token.
            token_id: ID del token que provocó la transición.

        Returns:
            None.

        Raises:
            GenerationError: Si la transición de estados no es válida o no
                puede ser sincronizada.
        """
        if (
            previous_state.phase == DecodingState.EXPECT_FN_NAME
            and state.phase == DecodingState.EXPECT_ARGS_KEY
        ):
            input_ids.extend(
                self._encode_fixed_text('", "args": {"')
            )

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

        generated_tokens = 0
        MAX_TOKENS = 200

        while state.phase != DecodingState.DONE:
            if generated_tokens >= MAX_TOKENS:
                raise GenerationError(
                    f"Número máximo de tokens generados superado: {MAX_TOKENS}"
                )

            logits = self._llm.get_logits(
                input_ids
            )

            constrained_logits = self._decoder.apply_constraints(
                logits,
                state,
            )

            if not np.any(np.isfinite(constrained_logits)):
                raise GenerationError(
                    "No hay tokens válidos para el estado actual"
                )

            next_token_id = int(
                np.argmax(constrained_logits)
            )

            input_ids.append(
                next_token_id
            )

            previous_state = state

            state = self._decoder.update_state(
                next_token_id,
                state,
            )

            self._sync_states(
                input_ids,
                previous_state,
                state,
            )

            generated_tokens += 1
