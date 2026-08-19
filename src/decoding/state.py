from enum import Enum
from pydantic import BaseModel


class DecodingState(Enum):
    """
    Representa los estados posibles durante a la generación restringida.
    """

    EXPECT_FN_NAME = "expect_fn_name"
    EXPECT_ARGS_KEY = "expect_args_key"
    EXPECT_ARGS_VALUE = "expect_args_value"
    EXPECT_SEPARATOR = "expect_separator"
    DONE = "done"


class DecoderState(BaseModel):
    """
    Representa el contexto dinámico de la máquina de estados durante
    la decodificación restringida.
    """

    phase: DecodingState
    selected_function: str | None = None
    current_parameter: str | None = None
    prefix: str = ""
