from llm_sdk.llm_sdk import Small_LLM_Model
from .models import FunctionFormat
from .token_cache import encode_ids
from .parameter_value_detail import (
    parameter_value_number,
    parameter_value_string,
    parameter_value_integer
)


def mask_allow_token(
    logits: list[float],
    allowed_ids: list[int]
) -> list[float]:
    """Mask all tokens except the explicitly allowed token IDs.

    Args:
        logits: Scores for every token in the model vocabulary.
        allowed_ids: Token IDs that are allowed to remain selectable.

    Returns:
        The logits with all disallowed tokens set to negative infinity.

    """

    for token_id in range(len(logits)):
        if token_id not in allowed_ids:
            logits[token_id] = float("-inf")
    return logits


def mask_fixed_sequence(
    logits: list[float],
    generated_ids: list[int],
    state_start_position: int,
    expected_ids: list[int]
) -> list[float]:
    """Allow only the next token of a predefined token sequence.
    The current position in the sequence is calculated from the number
    of tokens generated since entering the current state.

    Args:
        logits: Scores for every token in the model vocabulary.
        generated_ids: Token IDs generated for the current output.
        state_start_position: Position where the current state started.
        expected_ids: Token IDs of the fixed sequence to generate.

    Returns:
        The logits with only the expected next token left selectable.

    """

    current_position = len(generated_ids) - state_start_position
    for token_id in range(len(logits)):
        if token_id != expected_ids[current_position]:
            logits[token_id] = float("-inf")
    return logits


def mask_function_name(
    logits: list[float],
    function_generated_ids: list[int],
    function_names_ids: list[list[int]]
) -> list[float]:
    """Restrict generation to valid function-name continuations.
    Function names whose token prefix does not match the tokens already
    generated are discarded.

    Args:
        logits: Scores for every token in the model vocabulary.
        function_generated_ids: Function-name tokens generated so far.
        function_names_ids: Tokenized names of all available functions.

    Returns:
        The logits restricted to valid next function-name tokens.

    """

    allowed_function_ids: list[int] = []
    current_position = len(function_generated_ids)

    for function_name_ids in function_names_ids:
        is_matching = True
        for position in range(current_position):
            if position >= len(function_name_ids):
                is_matching = False
                break
            if function_generated_ids[position] != function_name_ids[position]:
                is_matching = False
                break
        if is_matching is False:
            continue
        if current_position >= len(function_name_ids):
            continue

        next_function_token_id = function_name_ids[current_position]
        if next_function_token_id not in allowed_function_ids:
            allowed_function_ids.append(next_function_token_id)

    return mask_allow_token(logits, allowed_function_ids)


def mask_parameter_name(
    logits: list[float],
    model: Small_LLM_Model,
    selected_function: FunctionFormat | None,
    parameter_generated_ids: list[int],
    completed_parameters: list[str],
) -> list[float]:
    """Restrict generation to parameter names of the selected function.
    Parameter names are tokenized and compared with the prefix generated
    so far.

    Args:
        logits: Scores for every token in the model vocabulary.
        model: Language model used to encode parameter names.
        selected_function: Function whose parameter name is being generated.
        parameter_generated_ids: Parameter-name tokens generated so far.

    Returns:
        The logits restricted to valid next parameter-name tokens.

    """
    if selected_function is None:
        return logits

    parameter_names_ids: list[list[int]] = []
    for parameter_name in selected_function.parameters:
        if parameter_name in completed_parameters:
            continue
        parameter_name_ids = encode_ids(model, f'"{parameter_name}"')
        parameter_names_ids.append(parameter_name_ids)

    allowed_parameter_ids: list[int] = []
    current_position = len(parameter_generated_ids)

    for parameter_name_ids in parameter_names_ids:
        is_matching = True
        for position in range(current_position):
            if position >= len(parameter_name_ids):
                is_matching = False
                break
            if (
                parameter_generated_ids[position]
                != parameter_name_ids[position]
            ):
                is_matching = False
                break
        if not is_matching:
            continue
        if current_position >= len(parameter_name_ids):
            continue
        next_parameter_token_id = parameter_name_ids[current_position]
        if next_parameter_token_id not in allowed_parameter_ids:
            allowed_parameter_ids.append(next_parameter_token_id)
    return mask_allow_token(logits, allowed_parameter_ids)


def mask_parameter_value(
    logits: list[float],
    model: Small_LLM_Model,
    selected_function: FunctionFormat | None,
    selected_parameters: str | None,
    value_generated_ids: list[int],
    comma_ids: list[int],
    closing_brace_ids: list[int],
    completed_parameters: list[str],
) -> list[float]:
    """Apply value constraints based on the selected parameter type.
    The selected parameter definition is inspected to determine its type.

    Args:
        logits: Scores for every token in the model vocabulary.
        model: Language model used for token encoding and decoding.
        selected_function: Function currently being generated.
        selected_parameter: Parameter currently being generated.
        value_generated_ids: Value tokens generated so far.
        comma_ids: Token IDs representing a comma.
        closing_brace_ids: Token IDs representing a closing brace.

    Returns:
        The logits after applying constraints for the parameter type.

    """
    if selected_function is None:
        return logits
    if selected_parameters is None:
        return logits

    parameter_info = selected_function.parameters[selected_parameters]
    parameter_type = parameter_info.type
    value_text = model.decode(value_generated_ids)

    if parameter_type == "number":
        return parameter_value_number(
            logits, model,
            value_text,
            comma_ids,
            closing_brace_ids,
            selected_function,
            selected_parameters,
            completed_parameters
        )
    elif parameter_type == "string":
        return parameter_value_string(
            logits,
            model,
            value_text,
            selected_parameters
        )

    elif parameter_type == "integer":
        return parameter_value_integer(
            logits,
            model,
            value_text,
            comma_ids,
            closing_brace_ids,
            selected_function,
            selected_parameters,
            completed_parameters,
        )

    return logits
