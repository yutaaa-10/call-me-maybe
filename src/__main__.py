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


def constrained_decoding(
    logits: list[float],
    model: Small_LLM_Model,
    state: State,
    function_list: list[FunctionFormat],
    generated_ids: list[int],
    function_generated_ids: list[int],
    state_start_position: int,
    selected_function: FunctionFormat | None,
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
        return logits


    # elif state == State.END:

    return logits


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
        state = State.START
        state_start_position = 0
        selected_function = None


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
                value_generated_ids.append(next_token_id)

                completed_parameters: list[str] = []
                if selected_function is not None and selected_parameter is not None:
                    if value_is_complete:
                        completed_parameters.append(selected_parameter)

                        if len(completed_parameters) == len(selected_function.parameters):
                            state = State.PARAMETER_SEPARATOR
                        else:
                            state = State.PARAMETER_NAME
                        state_start_position = len(generated_ids)

            elif state == State.PARAMETER_SEPARATOR:

            elif state == State.PARAMETERS_END:

            elif state ==State.END:




if __name__ == "__main__":
    main()
