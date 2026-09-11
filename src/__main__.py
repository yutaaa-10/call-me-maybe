from llm_sdk.llm_sdk import Small_LLM_Model
from .preparation import (
    open_file,
    check_prompts,
    check_functions,
    build_functions_text,
    build_context,
    FunctionFormat
)

from enum import Enum


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
        function_name_ids = model.encode(function.name)[0].tolist()
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
            parameter_name_ids = model.encode(parameter_name)[0].tolist()
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
                    if token_text.startswith(","):
                        starts_separator = True
                    if token_text.startswith("}"):
                        starts_separator = True
                if not can_continue_number and not starts_separator:
                    logits[token_id] = float("-inf")
            return logits

        elif parameter_type == "string":
            for token_id in range(len(logits)):
                token_text = model.decode([token_id])
                candidate_value = value_text + token_text
                if not is_valid_string_prefix(candidate_value):
                    logits[token_id] = float("-inf")
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

def value_is_complete(
        value_generated_ids: list[int],
        selected_function: FunctionFormat | None,
        selected_parameter: str,
        model: Small_LLM_Model,
        next_token_id: int,
    ) -> bool:

    if not value_generated_ids:
        return False

    if selected_function is None:
        return False

    if selected_parameter is None:
        return False

    parameter_info = selected_function.parameters[selected_parameter]
    parameter_type = parameter_info.type

    value_text = model.decode(value_generated_ids)
    next_token_text = model.decode([next_token_id])

    if parameter_type == "number":
        try:
            float(value_text)
        except ValueError:
            return False
        if next_token_text.startswith(","):
            return True

        if next_token_text.startswith("}"):
            return True
        return False

    if parameter_type == "string":
        if value_text.startswith('"') and value_text.endswith('"'):
            return True
        return False

    return False



def main() -> None:
    """Run the function-calling generation workflow.

    Load and validate prompts and function definitions, initialize the

    language model, prepare the available function information, and

    process each prompt to generate function calls.

    """

    prompts = open_file("data/input/function_calling_tests.json")
    prompt_list = check_prompts(prompts)

    functions = open_file("data/input/functions_definition.json")
    functions_list = check_functions(functions)

    model = Small_LLM_Model()
    functions_text = build_functions_text(functions_list)


    for prompt in prompt_list:
        encode_context = build_context(prompt, functions_text)
        token_ids = model.encode(encode_context)[0].tolist()

        generated_ids: list[int] = []
        function_generated_ids: list[int] = []
        parameter_generated_ids: list[int] = []
        value_generated_ids: list[int] = []
        completed_parameters: list[str] = []
        state = State.START
        state_start_position = 0
        selected_function = None
        selected_parameter = None


        while True:
            logits = model.get_logits_from_input_ids(token_ids)
            masked_logits = constrained_decoding(
                logits,
                model,
                state,
                functions_list,
                generated_ids,
                function_generated_ids,
                state_start_position,
                selected_function,
                selected_parameter,
                parameter_generated_ids,
                value_generated_ids,
            )

            max_logit = masked_logits[0]
            next_token_id = 0

            for token_id in range(len(masked_logits)):
                current_logit = masked_logits[token_id]
                if current_logit > max_logit:
                    max_logit = current_logit
                    next_token_id = token_id
            token_ids.append(next_token_id)
            generated_ids.append(next_token_id)



            if state == State.START:
                state = State.FUNCTION_KEY
                state_start_position = len(generated_ids)

            elif state == State.FUNCTION_KEY:
                function_ids = model.encode('"name":')[0].tolist()
                state_generated_count = len(generated_ids) - state_start_position

                if state_generated_count == len(function_ids):
                    state = State.FUNCTION_NAME
                    state_start_position = len(generated_ids)

            elif state == State.FUNCTION_NAME:
                function_generated_ids.append(next_token_id)

                for function in functions_list:
                    function_name_ids = model.encode(function.name)[0].tolist()
                    if function_generated_ids == function_name_ids:
                        selected_function = function
                        state = State.FUNCTION_SEPARATOR
                        state_start_position = len(generated_ids)
                        break

            elif state == State.FUNCTION_SEPARATOR:
                state = State.PARAMETERS_KEY
                state_start_position = len (generated_ids)

            elif state == State.PARAMETERS_KEY:
                parameter_ids = model.encode('"parameters":')[0].tolist()
                state_generated_count = len(generated_ids) - state_start_position

                if state_generated_count == len(parameter_ids):
                    state = State.PARAMETERS_START
                    state_start_position = len(generated_ids)

            elif state == State.PARAMETERS_START:
                state = State.PARAMETER_NAME
                state_start_position = len(generated_ids)

            elif state == State.PARAMETER_NAME:
                parameter_generated_ids.append(next_token_id)

                if selected_function is not None:
                    for parameter_name in selected_function.parameters:
                        parameter_name_ids = model.encode(parameter_name)[0].tolist()

                        if parameter_generated_ids == parameter_name_ids:
                            selected_parameter = parameter_name
                            state = State.PARAMETER_COLON
                            state_start_position = len(generated_ids)
                            break

            elif state == State.PARAMETER_COLON:
                state = State.PARAMETER_VALUE
                state_start_position = len(generated_ids)

            elif state == State.PARAMETER_VALUE:
                if selected_function is not None and selected_parameter is not None:
                    if value_is_complete(
                        value_generated_ids,
                        selected_function,
                        selected_parameter,
                        model,
                        next_token_id
                    ):
                        completed_parameters.append(selected_parameter)

                        if len(completed_parameters) == len(selected_function.parameters):

                            state = State.PARAMETERS_END
                        else:
                            state = State.PARAMETER_SEPARATOR
                        state_start_position = len(generated_ids)
                    else:
                        value_generated_ids.append(next_token_id)

            elif state == State.PARAMETER_SEPARATOR:
                state = State.PARAMETER_NAME
                parameter_generated_ids = []
                value_generated_ids = []
                selected_parameter = None
                state_start_position = len(generated_ids)

            elif state == State.PARAMETERS_END:
                state = State.END
                state_start_position = len(generated_ids)

            elif state == State.END:
                break

    aaa = model.decode(generated_ids)
    print(aaa)
    return logits



if __name__ == "__main__":
    main()
