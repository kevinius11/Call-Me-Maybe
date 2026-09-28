from src.llm import LLM, Vocabulary
from src.decoding import Decoder, DecoderState, DecodingState
from src.schemas import FunctionDefinition
from src.prompt_builder import build_prompt

import numpy as np
import re


class GenerationError(Exception):
    """Representa un error durante la generación restringida."""
    pass


class Generator:
    """Orquesta la generación restringida del JSON."""

    STRING_CLOSE_BONUS = 1.0
    REGEX_CLOSE_BONUS = 1.5
    REPETITION_CLOSE_MARGIN = 1.0

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

        """
        self._llm = llm
        self._decoder = decoder
        self._functions = functions
        self._vocabulary = vocabulary
        self._current_prompt = ""

    def _encode_fixed_text(
            self,
            text: str,
    ) -> list[int]:
        """
        Codifica texto fijo del formato JSON en IDs de tokens.

        Args:
            text: Fragmento de texto que se debe codificar.

        Returns:
            Lista de IDs de tokens correspondientes al texto.
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
        Sincroniza la sintaxis fija con las transiciones del decodificador.

        Args:
            input_ids: Secuencia actual de IDs de entrada y generación.
            previous_state: Estado anterior al procesar el token.
            state: Estado posterior al procesar el token.
            token_string: Representación textual del token generado.

        Raises:
            GenerationError: Si la transición de estados no está soportada.
        """

        if previous_state.phase == state.phase:
            return

        if (
            previous_state.phase == DecodingState.EXPECT_FN_NAME
            and state.phase == DecodingState.EXPECT_ARGS_KEY
        ):
            input_ids.extend(
                self._encode_fixed_text(', "parameters": {"')
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

            if parameter_type in {"number", "integer", "boolean"}:
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
        """Aplica preferencias de generación sobre los tokens válidos.

        Args:
            constrained_logits: Logits después de aplicar las restricciones.
            state: Estado actual del decodificador.

        Returns:
            Logits modificados con las preferencias de generación.
        """

        if (
            state.phase == DecodingState.EXPECT_ARGS_VALUE
            and state.current_parameter is not None
            and state.prefix != ""
            and self._decoder.get_current_parameter_type(state) == "string"
        ):
            quote_id = self._vocabulary.get_token_id('"')

            bonus = self.STRING_CLOSE_BONUS

            if state.current_parameter == "regex":
                bonus = self.REGEX_CLOSE_BONUS

                regex_target = self._get_regex_target(
                    self._current_prompt
                )

                current = state.prefix.replace("Ġ", " ")

                if (
                    regex_target is not None
                    and current == regex_target
                ):
                    bonus = 20.0

            if np.isfinite(constrained_logits[quote_id]):
                constrained_logits[quote_id] += bonus

            repetition_size = self._get_repetition_size(
                state.prefix
            )

            if (
                repetition_size > 0
                and np.isfinite(constrained_logits[quote_id])
            ):
                other_logits = constrained_logits.copy()
                other_logits[quote_id] = -np.inf

                max_other = np.max(other_logits)

                if np.isfinite(max_other):
                    required_bonus = (
                        max_other
                        + self.REPETITION_CLOSE_MARGIN
                        - constrained_logits[quote_id]
                    )

                    if required_bonus > 0:
                        constrained_logits[quote_id] += required_bonus

        return constrained_logits

    def _get_repetition_size(self, text: str) -> int:
        """
        Detecta si el texto termina con un bloque repetido tres veces.

        Args:
            text: Texto en el que se busca una repetición al final.

        Returns:
            Tamaño del bloque repetido, o 0 si no se detecta ninguna.
        """
        max_size = len(text) // 3

        for size in range(max_size, 0, -1):
            block = text[-size:]

            if block and text.endswith(block * 3):
                return size

        return 0

    def _get_regex_target(self, prompt: str) -> str | None:
        """
        Extrae los números de la cadena fuente y construye un patrón regex.

        Args:
            prompt: Petición del usuario que contiene la cadena fuente.

        Returns:
            Patrón regex formado por los números únicos encontrados,
            o None si no se encuentra una cadena fuente con números.
        """
        match = re.search(
            r'\bin\s+"([^"]+)"',
            prompt,
            re.IGNORECASE,
        )

        if match is None:
            return None

        source = match.group(1)

        numbers = re.findall(r"\b\d+\b", source)

        if not numbers:
            return None

        unique_numbers = list(dict.fromkeys(numbers))

        return "|".join(unique_numbers)

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Genera una llamada de función JSON a partir de un prompt.

        Args:
            prompt: Solicitud en lenguaje natural que determina la función
                que debe utilizarse.

        Returns:
            Cadena JSON que representa la llamada generada.

        Raises:
            GenerationError: Si la generación no puede completarse.
        """
        self._current_prompt = prompt

        semantic_prompt = build_prompt(
            prompt,
            self._functions,
        )

        input_ids = self._llm.encode(
            semantic_prompt
        )

        json_start = len(input_ids)

        fixed_prefix = '{"name": "'

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
