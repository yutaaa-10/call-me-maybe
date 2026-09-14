from llm_sdk.llm_sdk import Small_LLM_Model
from .models import FunctionFormat

def is_complete_number(text: str) -> bool:
    """Check whether text is a complete valid number."""

    try:
        float(text)
    except ValueError:
        return False

    return True

def is_valid_number_prefix(text: str) -> bool:
    """Check whether text can still become a valid number."""
    if text == "":
        return True
    if text == "-":
        return True
    if text.count(".") > 1:
        return False
    start_index = 0
    if text[0] == "-":
        start_index = 1
    for character in text[start_index:]:
        if character != "." and not character.isdigit():
            return False
    return True

def is_valid_string_prefix(text: str) -> bool:
    if text == "":
        return True
    if not text.startswith('"'):
        return False
    quote_count = text.count('"')
    if quote_count > 2:
        return False
    if quote_count == 2:
        if not text.endswith('"'):
            return False
    return True

def value_is_complete(
        value_generated_ids: list[int],
        selected_function: FunctionFormat | None,
        selected_parameter: str,
        model: Small_LLM_Model,
    ) -> bool:

    if not value_generated_ids:
        return False

    if selected_function is None:
        return False

    parameter_info = selected_function.parameters[selected_parameter]
    parameter_type = parameter_info.type

    value_text = model.decode(value_generated_ids)

    if parameter_type == "number":
        try:
            float(value_text)
            return True
        except ValueError:
            return False
    if parameter_type == "string":
        if len(value_text) < 3:
            return False
        if not value_text.startswith('"'):
            return False
        if not value_text.endswith('"'):
            return False
        if value_text.count('"') != 2:
            return False
        return True

