import sys
import argparse
from pathlib import Path
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .models import State
from .decoder import constrained_decoding
from .preparation import (
    open_file,
    check_prompts,
    check_functions,
    build_functions_text,
    build_context,
)
from .state_handler import (
    handle_function_name,
    handle_parameter_name,
    handle_parameter_value
)
from .token_cache import encode_ids
from .models import State, FunctionFormat


def prepare_inputs() -> (
    tuple[list[str], list[FunctionFormat], Path] | None
):
    """Parse CLI arguments and load validated inputs.

    Returns:
        Prompts, function definitions, and the output path.
        Returns None if input loading or validation fails.
    """
    parser = argparse.ArgumentParser(
        description="Generate function calls from natural-language prompts."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    prompts = open_file(args.input)
    if prompts is None:
        return None

    functions = open_file(args.functions_definition)
    if functions is None:
        return None

    prompt_list = check_prompts(prompts)
    if prompt_list is None:
        return None

    functions_list = check_functions(functions)
    if functions_list is None:
        return None

    return prompt_list, functions_list, Path(args.output)


def main() -> int:
    """Run the function-calling generation workflow.
    Load and validate prompts and function definitions, initialize the
    language model, prepare the available function information, and
    process each prompt to generate function calls.

    """

    prepared = prepare_inputs()
    if prepared is None:
        return 1
    prompt_list, functions_list, output_path = prepared

    model = Small_LLM_Model()
    functions_text = build_functions_text(functions_list)
    results: list[dict[str, object]] = []

    MAX_GENERATION_STEPS = 200

    for prompt in prompt_list:
        print(prompt)
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
        selected_parameters = None

        generation_count = 0

        while True:
            generation_count += 1
            if generation_count > MAX_GENERATION_STEPS:
                print(
                    f"Error: Generation limit reached for prompt {prompt!r}",
                    file=sys.stderr,
                )
                return 1
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
                selected_parameters,
                parameter_generated_ids,
                value_generated_ids,
                completed_parameters,
            )
            if all(logit == float("-inf") for logit in masked_logits):
                print(
                    f"Error: No valid token available in state {state.value}.",
                    file=sys.stderr,
                )
                return 1

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
                function_ids = encode_ids(model, '"name":')
                state_generated_count = len(
                    generated_ids) - state_start_position

                if state_generated_count == len(function_ids):
                    state = State.FUNCTION_NAME
                    state_start_position = len(generated_ids)

            elif state == State.FUNCTION_NAME:
                (
                    state,
                    selected_function,
                    state_start_position,
                ) = handle_function_name(
                    model,
                    function_generated_ids,
                    functions_list,
                    generated_ids,
                    next_token_id,
                )

            elif state == State.FUNCTION_SEPARATOR:
                state = State.PARAMETERS_KEY
                state_start_position = len(generated_ids)

            elif state == State.PARAMETERS_KEY:
                parameter_ids = encode_ids(model, '"parameters":')
                state_generated_count = len(
                    generated_ids) - state_start_position

                if state_generated_count == len(parameter_ids):
                    state = State.PARAMETERS_START
                    state_start_position = len(generated_ids)

            elif state == State.PARAMETERS_START:
                state = State.PARAMETER_NAME
                state_start_position = len(generated_ids)

            elif state == State.PARAMETER_NAME:
                (
                    state,
                    selected_parameters,
                    state_start_position,
                ) = handle_parameter_name(
                    model,
                    parameter_generated_ids,
                    selected_function,
                    generated_ids,
                    next_token_id,
                )

            elif state == State.PARAMETER_COLON:
                state = State.PARAMETER_VALUE
                state_start_position = len(generated_ids)

            elif state == State.PARAMETER_VALUE:
                (
                    state,
                    selected_parameters,
                    parameter_generated_ids,
                    value_generated_ids,
                    state_start_position,
                ) = handle_parameter_value(
                    model,
                    selected_function,
                    selected_parameters,
                    next_token_id,
                    generated_ids,
                    parameter_generated_ids,
                    value_generated_ids,
                    completed_parameters,
                )

            elif state == State.PARAMETER_SEPARATOR:
                state = State.PARAMETER_NAME
                parameter_generated_ids = []
                value_generated_ids = []
                selected_parameters = None
                state_start_position = len(generated_ids)

            elif state == State.PARAMETERS_END:
                state = State.END
                state_start_position = len(generated_ids)

            elif state == State.END:
                generated_text = model.decode(generated_ids)
                try:
                    generated_result = json.loads(generated_text)
                except json.JSONDecodeError as e:
                    print(
                        f"Error: Generated invalid "
                        f"JSON for prompt {prompt!r}: {e}",
                        file=sys.stderr,
                    )
                    return 1

                result = {
                    "prompt": prompt,
                    "name": generated_result["name"],
                    "parameters": generated_result["parameters"],
                }
                results.append(result)
                break

    output_path = Path(args.output)

    try:
        output_text = json.dumps(
            results,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as e:
        print(
            f"Error: Cannot save output to {output_path}: {e}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
