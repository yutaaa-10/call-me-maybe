from .models import State, FunctionFormat
from llm_sdk.llm_sdk import Small_LLM_Model
from .decoder_mask import (
    mask_allow_token,
    mask_fixed_sequence,
    mask_function_name,
    mask_parameter_name,
    mask_parameter_value,
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
        return mask_allow_token(logits, braces_id)

    elif state == State.FUNCTION_KEY:
        return mask_fixed_sequence(logits, generated_ids, state_start_position, name_ids)

    elif state == State.FUNCTION_NAME:
        return mask_function_name(logits, function_generated_ids, function_names_ids)

    elif state == State.FUNCTION_SEPARATOR:
        return mask_allow_token(logits, comma_ids)

    elif state == State.PARAMETERS_KEY:
        return mask_fixed_sequence(logits, generated_ids, state_start_position, parameters_ids)

    elif state == State.PARAMETERS_START:
        return mask_allow_token(logits, braces_id)

    elif state == State.PARAMETER_NAME:
        return mask_parameter_name(logits, model, selected_function, parameter_generated_ids)

    elif state == State.PARAMETER_COLON:
        return mask_allow_token(logits, colon_ids)

    elif state == State.PARAMETER_VALUE:
        return mask_parameter_value(
            logits, model,
            selected_function,
            selected_parameter,
            value_generated_ids,
            comma_ids,
            closing_brace_ids
        )

    elif state == State.PARAMETER_SEPARATOR:
        return mask_allow_token(logits, comma_ids)

    elif state == State.PARAMETERS_END:
        return mask_allow_token(logits, closing_brace_ids)

    elif state == State.END:
        return mask_allow_token(logits, closing_brace_ids)

    return logits
