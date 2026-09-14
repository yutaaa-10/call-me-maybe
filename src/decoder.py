from .models import State, FunctionFormat
from llm_sdk.llm_sdk import Small_LLM_Model
from .validators import (
    is_valid_number_prefix,
    is_complete_number
)


def constrained_decoding(
    logits: list[float],
    model: Small_LLM_Model,
    state: State,
    function_list: list[FunctionFormat],
    generated_ids: list[int],
    function_generated_ids: list[int],
    state_start_position: int,
    selected_function: FunctionFormat | None,
    selected_parameter: str | None,
    parameter_generated_ids: list[int],
    value_generated_ids: list[int],
) -> list[float]:

    braces_id = model.encode("{")[0].tolist()
    comma_ids = model.encode(",")[0].tolist()
    colon_ids = model.encode(":")[0].tolist()

    name_ids = model.encode('"name":')[0].tolist()
    parameters_ids = model.encode('"parameters":')[0].tolist()

    function_names_ids: list[list[int]] = []
    for function in function_list:
        function_name_ids = model.encode(f'"{function.name}"')[0].tolist()
        function_names_ids.append(function_name_ids)

    closing_brace_ids = model.encode("}")[0].tolist()


    if state == State.START:
        for token_id in range(len(logits)):
            if token_id not in braces_id:
                logits[token_id] = float("-inf")
        return logits

    elif state == State.FUNCTION_KEY:
        current_position = len(generated_ids) - state_start_position
        for token_id in range(len(logits)):
            if token_id != name_ids[current_position]:
                logits[token_id] = float("-inf")
        return logits

    elif state == State.FUNCTION_NAME:
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

        for token_id in range(len(logits)):
            if token_id not in allowed_function_ids:
                logits[token_id] = float("-inf")

        return logits

    elif state == State.FUNCTION_SEPARATOR:
        for token_id in range(len(logits)):
            if token_id not in comma_ids:
                logits[token_id] = float("-inf")
        return logits


    elif state == State.PARAMETERS_KEY:
        current_position = len(generated_ids) - state_start_position
        next_parameter_token_id = parameters_ids[current_position]
        for token_id in range(len(logits)):
            if token_id != next_parameter_token_id:
                logits[token_id] = float("-inf")
        return logits

    elif state == State.PARAMETERS_START:
        for token_id in range(len(logits)):
            if token_id not in braces_id:
                logits[token_id] = float("-inf")
        return logits


    elif state == State.PARAMETER_NAME:

        if selected_function is None:
            return logits
        parameter_names_ids: list[list[int]] = []
        for parameter_name in selected_function.parameters:
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
                if parameter_generated_ids[position] != parameter_name_ids[position]:
                    is_matching = False
                    break
            if is_matching is False:
                continue
            if current_position >= len(parameter_name_ids):
                continue
            next_parameter_token_id = parameter_name_ids[current_position]
            if next_parameter_token_id not in allowed_parameter_ids:
                allowed_parameter_ids.append(next_parameter_token_id)

        for token_id in range(len(logits)):
            if token_id not in allowed_parameter_ids:
                logits[token_id] = float("-inf")
        return logits

    elif state == State.PARAMETER_COLON:
        for token_id in range(len(logits)):
            if token_id not in colon_ids:
                logits[token_id] = float("-inf")
        return logits

    elif state == State.PARAMETER_VALUE:
        if selected_function is None:
            return logits
        if selected_parameter is None:
            return logits
        parameter_info = selected_function.parameters[selected_parameter]
        parameter_type = parameter_info.type

        value_text = model.decode(value_generated_ids)

        if parameter_type == "number":
            number_is_complete = is_complete_number(value_text)
            for token_id in range(len(logits)):
                token_text = model.decode([token_id])
                candidate_value = value_text + token_text
                can_continue_number = is_valid_number_prefix(candidate_value)
                starts_separator = False
                if number_is_complete:
                    if token_id in comma_ids:
                        starts_separator = True
                    if token_id in closing_brace_ids:
                        starts_separator = True
                if not can_continue_number and not starts_separator:
                    logits[token_id] = float("-inf")
            return logits

        elif parameter_type == "string":
            quote_ids = model.encode('"')[0].tolist()
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
                if "{" in token_text or "}" in token_text:
                    logits[token_id] = float("-inf")
                    continue
                if '\\"' in token_text:
                    logits[token_id] = float("-inf")
                    continue
                if candidate_value == '""':
                    logits[token_id] = float("-inf")
                    continue
                if candidate_value.count('"') > 2:
                    logits[token_id] = float("-inf")
                    continue
                if candidate_value.count('"') == 2:
                    if not candidate_value.endswith('"'):
                        logits[token_id] = float("-inf")
                        continue
            if len(value_text) > 1:

                for quote_id in quote_ids:

                    if logits[quote_id] != float("-inf"):

                        logits[quote_id] += 4.0
            return logits

    elif state == State.PARAMETER_SEPARATOR:
        for token_id in range(len(logits)):
            if token_id not in comma_ids:
                logits[token_id] = float("-inf")
        return logits

    elif state == State.PARAMETERS_END:
            for token_id in range(len(logits)):
                if token_id not in closing_brace_ids:
                    logits[token_id] = float("-inf")
            return logits

    elif state == State.END:
            for token_id in range(len(logits)):
                if token_id not in closing_brace_ids:
                    logits[token_id] = float("-inf")
            return logits
    return logits

