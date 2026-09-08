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
    FUNCTION_KEY = "name_key"
    FUNCTION_NAME = "function_name"
    PARAMETERS_KEY = "parameters_key"
    PARAMETER_NAME = "parameter_name"
    PARAMETER_VALUE = "parameter_value"
    END = "end"


def constrained_decoding(
    logits: list[float],
    model: Small_LLM_Model,
    state: State,
    function_list: list[FunctionFormat],
    generated_ids: list[int],
    function_generated_ids: list[int],
    state_start_position: int
) -> list[float]:

    allowed_ids = model.encode("{")[0].tolist()
    name_ids = model.encode('"name":')[0].tolist()
    parameters_ids = model.encode('"parameters":')[0].tolist()

    function_names_ids: list[list[int]] = []
    for function in function_list:
        function_name_ids = model.encode(function.name)[0].tolist()
        function_names_ids.append(function_name_ids)



    if state == State.START:
        for token_id in range(len(logits)):
            if token_id not in allowed_ids:
                logits[token_id] = float("-inf")
        return logits

    elif state == State.FUNCTION_KEY:
        current_position = len(generated_ids) - state_start_position
        next_name_token_id = name_ids[current_position]
        for token_id in range(len(logits)):
            if token_id != next_name_token_id:
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


    elif state == State.PARAMETERS_KEY:
        current_position = len(generated_ids) - state_start_position
        next_parameter_token_id = parameters_ids[current_position]
        for token_id in range(len(logits)):
            if token_id != next_parameter_token_id:
                logits[token_id] = float("-inf")
        return logits



    # elif state == State.PARAMETER_NAME:

    # elif state == State.PARAMETER_VALUE:

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
        state = State.START
        state_start_position = 0

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
                        state = State.PARAMETERS_KEY
                        state_start_position = len(generated_ids)

            elif state == State.PARAMETERS_KEY:
                parameter_ids = model.encode('"parameters":')[0].tolist()
                state_generated_count = len(generated_ids) - state_start_position

                if state_generated_count == len(parameter_ids):
                    state == State.PARAMETER_NAME
                    state_start_position = len(generated_ids)


            break


if __name__ == "__main__":
    main()
