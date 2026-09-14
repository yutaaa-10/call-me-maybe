from pydantic import BaseModel
from enum import Enum

class PromptData(BaseModel):
    prompt: str


class ParameterInfo(BaseModel):
    type: str


class FunctionFormat(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterInfo]
    returns: ParameterInfo

class State(Enum):
    START = "start"

    FUNCTION_KEY = "function_key"
    FUNCTION_NAME = "function_name"
    FUNCTION_SEPARATOR = "function_separator"

    PARAMETERS_KEY = "parameters_key"
    PARAMETERS_START = "parameters_start"

    PARAMETER_NAME = "parameter_name"
    PARAMETER_COLON = "parameter_colon"
    PARAMETER_VALUE = "parameter_value"
    PARAMETER_SEPARATOR = "parameter_separator"

    PARAMETERS_END = "parameters_end"
    END = "end"