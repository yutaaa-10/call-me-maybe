from .validators import (
    is_valid_number_prefix,
    is_complete_number,
    is_complete_regex,
    is_valid_integer_prefix,
    count_unescaped_quotes
)
from llm_sdk.llm_sdk import Small_LLM_Model
from .models import FunctionFormat
from .token_cache import encode_ids


def parameter_value_integer(
    logits: list[float],
    model: Small_LLM_Model,
    value_text: str,
    comma_ids: list[int],
    closing_brace_ids: list[int],
    selected_function: FunctionFormat,
    selected_parameter: str,
    completed_parameters: list[str],
) -> list[float]:
    """Apply integer value constraints during token generation.
    The current integer text is extended with each possible token from the
    model vocabulary.

    Args:
        logits: Scores for every token in the model vocabulary.
        model: Language model used for token decoding.
        value_text: Integer value text generated so far.
        comma_ids: Token IDs representing a comma.
        closing_brace_ids: Token IDs representing a closing brace.
        selected_function: Function currently being generated.
        selected_parameter: Parameter currently being generated.
        completed_parameters: Parameters that have already been generated.

    Returns:
        The logits after masking tokens that cannot form or finish a valid
        integer value.

    """

    integer_is_complete = (
        value_text != ""
        and value_text != "-"
        and value_text.lstrip("-").isdigit()
    )
    remaining_parameters = [
        parameter_name
        for parameter_name in selected_function.parameters
        if parameter_name not in completed_parameters
        and parameter_name != selected_parameter
    ]
    has_next_parameter = len(remaining_parameters) > 0
    for token_id in range(len(logits)):
        token_text = model.decode([token_id])
        candidate_value = value_text + token_text
        can_continue_integer = is_valid_integer_prefix(candidate_value)
        can_finish_integer = False
        if integer_is_complete:
            if has_next_parameter and token_id in comma_ids:
                can_finish_integer = True
            if not has_next_parameter and token_id in closing_brace_ids:
                can_finish_integer = True
        if not can_continue_integer and not can_finish_integer:
            logits[token_id] = float("-inf")
    return logits


def parameter_value_number(
    logits: list[float],
    model: Small_LLM_Model,
    value_text: str,
    comma_ids: list[int],
    closing_brace_ids: list[int],
    selected_function: FunctionFormat,
    selected_parameters: str,
    completed_parameters: list[str],
) -> list[float]:
    """Restrict generation to tokens that can form a valid number.
    Tokens that cannot continue the current numeric value are masked.

    Args:
        logits: Scores for every token in the model vocabulary.
        model: Language model used to decode candidate tokens.
        value_text: Numeric value generated so far.
        comma_ids: Token IDs representing a comma.
        closing_brace_ids: Token IDs representing a closing brace.

    Returns:
        The logits restricted to valid numeric continuations or endings.

    """

    number_is_complete = is_complete_number(value_text)

    number_can_finish = (
        number_is_complete
        and "." in value_text
        and not value_text.endswith(".")
    )

    remaining_parametrs = [
        parameter_name
        for parameter_name in selected_function.parameters
        if parameter_name not in completed_parameters
        and parameter_name != selected_parameters
    ]

    has_next_parameter = len(remaining_parametrs) > 0

    for token_id in range(len(logits)):
        token_text = model.decode([token_id])
        candidate_value = value_text + token_text
        can_continue_number = is_valid_number_prefix(candidate_value)
        can_finish_number = False

        if number_can_finish:
            if has_next_parameter and token_id in comma_ids:
                can_finish_number = True
            if not has_next_parameter and token_id in closing_brace_ids:
                can_finish_number = True

        if not can_continue_number and not can_finish_number:
            logits[token_id] = float("-inf")
    return logits


def parameter_value_string(
    logits: list[float],
    model: Small_LLM_Model,
    value_text: str,
    selected_parameter: str | None,
) -> list[float]:
    """Restrict generation to tokens that can form a valid JSON string.

    Args:
        logits: Scores for every token in the model vocabulary.
        model: Language model used to encode and decode tokens.
        value_text: String value generated so far.
        selected_parameter: Name of the parameter currently being generated.

    Returns:
        The logits restricted to valid string continuations.

    """
    quote_ids = encode_ids(model, '"')
    for token_id in range(len(logits)):
        token_text = model.decode([token_id])
        candidate_value = value_text + token_text
        if value_text == "":
            if token_id not in quote_ids:
                logits[token_id] = float("-inf")
            continue
        if not value_text.startswith('"'):
            logits[token_id] = float("-inf")
            continue
        if "\n" in token_text or "\r" in token_text:
            logits[token_id] = float("-inf")
            continue

        quote_count = count_unescaped_quotes(candidate_value)
        if candidate_value == '""':
            logits[token_id] = float("-inf")
            continue
        if quote_count > 2:
            logits[token_id] = float("-inf")
            continue
        if quote_count == 2:
            if not candidate_value.endswith('"'):
                logits[token_id] = float("-inf")
                continue
    if selected_parameter == "regex":
        if is_complete_regex(value_text):
            for quote_id in quote_ids:
                if logits[quote_id] != float("-inf"):
                    logits[quote_id] += 10.0
    elif len(value_text) > 1:
        for quote_id in quote_ids:
            if logits[quote_id] != float("-inf"):
                logits[quote_id] += 4.0
    return logits
