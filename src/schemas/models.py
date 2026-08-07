from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    """
    Representa la definición de tipo de un parámetro o valor de retorno.
    Attributes:
        type: Nombre del tipo JSON (ej. "number", "string", "boolean").
    """
    type: str


class FunctionDefinition(BaseModel):
    """
    Representa la definicion de tipo de funcion.
    Attributes:
        name: Nombre de la funcion
        description: Descripcion de la funcion
        parameters: Diccionario de nombre de parámetro a su
        ParameterDefinition.
        returns: Elemento de retorno
    """
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class FunctionCallResult(BaseModel):
    """
    Representa el resultado de una llamada a funcion.
    Attributes:
        prompt: Sera la entrega enviada a la funcion.
        fn_name: Sera el nombre de la funcion seleccionada
        args: Los argumentos pasados como parametros de la funcion.
    """
    prompt: str
    fn_name: str
    args: dict[str, float | str | bool]
