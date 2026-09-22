from .validators import (
    is_valid_number_prefix,
    is_complete_number,
    is_complete_regex,
)
from llm_sdk.llm_sdk import Small_LLM_Model
from .models import FunctionFormat


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
    generated are discarded. Only valid next tokens from the remaining
    function names are allowed.

    Args:
        logits: Scores for every token in the model vocabulary.
        function_generated_ids: Function-name tokens generated so far.
        function_names_ids: Tokenized names of all available functions.

    Returns:
        The logits restricted to valid next function-name tokens.

    """

    allowed_function_ids: list[int] = []
    current_position = len(function_generated_ids)

    # それぞれのfunctionのcurrnt positionを見ていく
    for function_name_ids in function_names_ids:
        is_matching = True
        for position in range(current_position):
            # 関数名がpositionよりも短い
            if position >= len(function_name_ids):
                is_matching = False
                break
            # 作ったきたfunction_generatedと違ったら
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
    so far. Only tokens that can continue a valid parameter name remain.

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
        parameter_name_ids = model.encode(f'"{parameter_name}"')[0].tolist()
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
    Once the current value is a complete number, a comma or closing
    brace is also allowed to terminate the value.

    Args:
        logits: Scores for every token in the model vocabulary.
        model: Language model used to decode candidate tokens.
        value_text: Numeric value generated so far.
        comma_ids: Token IDs representing a comma.
        closing_brace_ids: Token IDs representing a closing brace.

    Returns:
        The logits restricted to valid numeric continuations or endings.

    """

    # 数値として成り立つか
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
        # くっつけた時に数値かどうか
        can_continue_number = is_valid_number_prefix(candidate_value)
        can_finish_number = False

        if number_can_finish:
            # 次のparameterがある場合だけ "," を許可
            if has_next_parameter and token_id in comma_ids:
                can_finish_number = True
            # 最後のparameterなら "}" を許可
            if not has_next_parameter and token_id in closing_brace_ids:
                can_finish_number = True

        # 数値ではないかつ、,}でないなら負の無限大
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
    The string must begin with a double quote and must not contain tokens
    that would break the surrounding JSON structure. Once some content
    has been generated, the closing quote receives a small logit bonus
    to encourage the model to finish the string.

    Args:
        logits: Scores for every token in the model vocabulary.
        model: Language model used to encode and decode tokens.
        value_text: String value generated so far.

    Returns:
        The logits restricted to valid string continuations.

    """
    quote_ids = model.encode('"')[0].tolist()
    for token_id in range(len(logits)):
        token_text = model.decode([token_id])
        candidate_value = value_text + token_text
        # まだ何も生成していない時は必ず"にする
        if value_text == "":
            if token_id not in quote_ids:
                logits[token_id] = float("-inf")
            continue
        if not value_text.startswith('"'):
            logits[token_id] = float("-inf")
            continue
        # 改行文字などの制御文字を禁止
        if "\n" in token_text or "\r" in token_text:
            logits[token_id] = float("-inf")
            continue
        # {,}も禁止にしている
        if "{" in token_text or "}" in token_text:
            logits[token_id] = float("-inf")
            continue
        # \\も判定が難しくなるので禁止
        if '\\"' in token_text:
            logits[token_id] = float("-inf")
            continue
        # "の次に"になることを禁止
        if candidate_value == '""':
            logits[token_id] = float("-inf")
            continue
        # "が三個以上は禁止
        if candidate_value.count('"') > 2:
            logits[token_id] = float("-inf")
            continue
        if candidate_value.count('"') == 2:
            if not candidate_value.endswith('"'):
                logits[token_id] = float("-inf")
                continue

    if selected_parameter == "regex":
        allowed_tokens: list[tuple[float, str, int]] = []
        for token_id in range(len(logits)):
            if logits[token_id] != float("-inf"):
                token_text = model.decode([token_id])
                allowed_tokens.append(
                    (logits[token_id], token_text, token_id)
                )
        allowed_tokens.sort(reverse=True)

    if selected_parameter == "regex":
        if is_complete_regex(value_text):
            for quote_id in quote_ids:
                if logits[quote_id] != float("-inf"):
                    logits[quote_id] += 10.0
    # 通常のstringは従来通り
    elif len(value_text) > 1:
        for quote_id in quote_ids:
            if logits[quote_id] != float("-inf"):
                logits[quote_id] += 4.0
    return logits


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
    Numeric parameters are handled by the number-value mask, while string
    parameters are handled by the string-value mask.

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

    # 全logitsを調べて「数値として続けられるToken」または,}以外を -inf にして生成候補から除外する。
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
    # 語彙にある全Tokenのlogitsを確認して、「次の文字列Tokenとして不正な候補」を -inf にして除外する
    elif parameter_type == "string":
        return parameter_value_string(
            logits,
            model,
            value_text,
            selected_parameters
        )

    return logits
