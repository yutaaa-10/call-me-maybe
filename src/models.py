from pydantic import BaseModel
from enum import Enum


class PromptData(BaseModel):
    """Represent and validate a user prompt."""
    prompt: str


class ParameterInfo(BaseModel):
    """Represent and validate information about a parameter type."""
    type: str


class FunctionFormat(BaseModel):
    """Represent and validate a function definition.
    A function definition contains its name, description,
    parameters, and return type.

    """
    name: str
    description: str
    parameters: dict[str, ParameterInfo]
    returns: ParameterInfo


class State(Enum):
    """Represent each state of the constrained JSON generation process."""

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
