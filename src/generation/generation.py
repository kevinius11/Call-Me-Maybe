from src.llm import LLM, Vocabulary
from src.decoding import Decoder, DecoderState, DecodingState
from src.schemas import FunctionDefinition
from src.prompt_builder import build_prompt

import numpy as np


class GenerationError(Exception):
    """Representa un error durante la generación restringida."""
    pass


class Generator:
    """Orquesta la generación restringida del JSON."""

    STRING_CLOSE_BONUS = 1.0

    def __init__(self,
                 llm: LLM,
                 decoder: Decoder,
                 functions: list[FunctionDefinition],
                 vocabulary: Vocabulary,
                 ) -> None:
        """
        Inicializa el generador.

        Args:
            llm: Modelo utilizado para generar tokens.
            decoder: Decoder encargado de restringir la generación.
            functions: Funciones disponibles para el modelo.
            vocabulary: Vocabulario utilizado para interpretar los tokens.

        Returns:
            None.
        """
        self._llm = llm
        self._decoder = decoder
        self._functions = functions
        self._vocabulary = vocabulary

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
        token_string: str,
    ) -> None:
        """
        Sincroniza la generación con la transición de estados actual.

        Args:
            input_ids: Lista de token IDs que representa el contexto actual
                de generación.
            previous_state: Estado de la máquina antes de procesar el token.
            state: Estado de la máquina después de procesar el token.
            token_string: Representación textual del token que provocó
                la transición.

        Returns:
            None.

        Raises:
            GenerationError: Si la transición de estados no es válida o no
                puede ser sincronizada.
        """

        if previous_state.phase == state.phase:
            return

        if (
            previous_state.phase == DecodingState.EXPECT_FN_NAME
            and state.phase == DecodingState.EXPECT_ARGS_KEY
        ):
            input_ids.extend(
                self._encode_fixed_text(', "args": {"')
            )
            return

        if (
            previous_state.phase == DecodingState.EXPECT_ARGS_KEY
            and state.phase == DecodingState.EXPECT_ARGS_VALUE
        ):
            parameter_type = self._decoder.get_current_parameter_type(
                state
            )

            if parameter_type == "string":
                input_ids.extend(
                    self._encode_fixed_text(': "')
                )
                return

            if parameter_type == "number":
                input_ids.extend(
                    self._encode_fixed_text(": ")
                )
                return

            raise GenerationError(
                f"Tipo de parámetro no soportado: {parameter_type}"
            )

        if (
            previous_state.phase == DecodingState.EXPECT_ARGS_VALUE
            and state.phase == DecodingState.EXPECT_ARGS_KEY
        ):
            if token_string == ",":
                input_ids.extend(
                    self._encode_fixed_text('"')
                )
                return

            if token_string == '"':
                input_ids.extend(
                    self._encode_fixed_text(', "')
                )
                return

        if (
            previous_state.phase == DecodingState.EXPECT_ARGS_VALUE
            and state.phase == DecodingState.DONE
        ):
            if token_string == "}":
                input_ids.extend(
                    self._encode_fixed_text("}")
                )
                return

            if token_string == '"':
                input_ids.extend(
                    self._encode_fixed_text("}}")
                )
                return

        raise GenerationError(
            "Transición de estados no soportada"
        )

    def _apply_generation_bias(
            self,
            constrained_logits: np.ndarray,
            state: DecoderState,
    ) -> np.ndarray:
        """
        Aplica preferencias de generación sobre los tokens válidos.

        Args:
            constrained_logits: Logits ya restringidos por el decoder.
            state: Estado actual de la máquina de decodificación.

        Returns:
            Logits modificados con la política de generación.
        """
        if (
            state.phase == DecodingState.EXPECT_ARGS_VALUE
            and state.current_parameter is not None
            and state.prefix != ""
            and self._decoder.get_current_parameter_type(state) == "string"
        ):
            quote_id = self._vocabulary.get_token_id('"')

            if np.isfinite(constrained_logits[quote_id]):
                constrained_logits[quote_id] += self.STRING_CLOSE_BONUS

        return constrained_logits

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Genera una llamada JSON restringida a partir de un prompt.

        Args:
            prompt: Solicitud en lenguaje natural que determina la función
                que debe ser invocada y sus argumentos.

        Returns:
            Cadena JSON que representa la llamada a una función.

        Raises:
            GenerationError: Si se supera el límite máximo de tokens
                o no existen tokens válidos para el estado actual.
        """
        semantic_prompt = build_prompt(
            prompt,
            self._functions,
        )

        input_ids = self._llm.encode(
            semantic_prompt
        )

        json_start = len(input_ids)

        fixed_prefix = '{"fn_name": "'

        input_ids.extend(
            self._encode_fixed_text(
                fixed_prefix
            )
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

            if not np.any(
                np.isfinite(constrained_logits)
            ):
                raise GenerationError(
                    "No hay tokens válidos para el estado actual"
                )

            constrained_logits = self._apply_generation_bias(
                constrained_logits,
                state,
            )

            next_token_id = int(
                np.argmax(constrained_logits)
            )

            token_string = self._vocabulary.get_token(
                next_token_id
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
                token_string,
            )

            generated_tokens += 1

        json_token_ids = input_ids[json_start:]

        return self._llm.decode(
            json_token_ids
        )
