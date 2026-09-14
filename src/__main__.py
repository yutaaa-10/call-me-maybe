from llm_sdk.llm_sdk import Small_LLM_Model
from .models import State
import json
from .preparation import (
    open_file,
    check_prompts,
    check_functions,
    build_functions_text,
    build_context,
)

from .validators import value_is_complete
from .decoder import constrained_decoding


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
    results: list[dict] = []


    for prompt in prompt_list:
        print("CURRENT PROMPT:", prompt)
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


        generation_count = 0
        while True:

            generation_count += 1

            if generation_count > 200:

                print("Generation limit reached")

                print("PROMPT:", prompt)

                print("STATE:", state)

                print("VALUE:", model.decode(value_generated_ids))

                break
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
                    function_name_ids = model.encode(
                        f'"{function.name}"'
                    )[0].tolist()
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
                        parameter_name_ids = model.encode(f'"{parameter_name}"')[0].tolist()

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
                    parameter_info = selected_function.parameters[selected_parameter]
                    parameter_type = parameter_info.type
                    next_token_text = model.decode([next_token_id])
                    if parameter_type == "number":
                        if next_token_text.startswith(","):
                            completed_parameters.append(selected_parameter)
                            parameter_generated_ids = []
                            value_generated_ids = []
                            selected_parameter = None
                            if next_token_text.startswith(',"'):
                                parameter_generated_ids = model.encode('"')[0].tolist()
                            state = State.PARAMETER_NAME
                            state_start_position = len(generated_ids)
                        elif next_token_text.startswith("}"):
                            completed_parameters.append(selected_parameter)
                            value_generated_ids = []
                            selected_parameter = None
                            state = State.END
                            state_start_position = len(generated_ids)
                        else:
                            value_generated_ids.append(next_token_id)
                    else:
                        value_generated_ids.append(next_token_id)
                        if value_is_complete(
                            value_generated_ids,
                            selected_function,
                            selected_parameter,
                            model,
                        ):
                            completed_parameters.append(selected_parameter)
                            if len(completed_parameters) == len(
                                selected_function.parameters
                            ):
                                state = State.PARAMETERS_END
                            else:
                                state = State.PARAMETER_SEPARATOR
                            state_start_position = len(generated_ids)


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
                generated_text = model.decode(generated_ids)
                generated_result = json.loads(generated_text)
                result = {
                    "prompt": prompt,
                    "name": generated_result["name"],
                    "parameters": generated_result["parameters"],
                }
                results.append(result)
                break

    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    return logits

if __name__ == "__main__":
    main()
