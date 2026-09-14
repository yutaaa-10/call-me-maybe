from llm_sdk.llm_sdk import Small_LLM_Model
from .models import State, FunctionFormat

def handle_function_name(
        model: Small_LLM_Model, 
        function_generated_ids: list[int], 
        functions_list: list[FunctionFormat], 
        generated_ids: list[int], 
        next_token_id: int
    ) -> tuple[State, FunctionFormat | None, int]:
    function_generated_ids.append(next_token_id)
    for function in functions_list:
        function_name_ids = model.encode(
            f'"{function.name}"'
        )[0].tolist()
        if function_generated_ids == function_name_ids:
            selected_function = function
            state = State.FUNCTION_SEPARATOR
            state_start_position = len(generated_ids)

            return(
                state,
                selected_function,
                state_start_position
            )

    return(
        State.FUNCTION_KEY,
        None,
        len(generated_ids)
    )

def handle_parameter_name(
    model: Small_LLM_Model,
    parameter_generated_ids: list[int],
    selected_function: FunctionFormat | None,
    generated_ids: list[int],
    next_token_id: int,
) -> tuple[State, str | None, int]:
    parameter_generated_ids.append(next_token_id)

    if selected_function is not None:
        for parameter_name in selected_function.parameters:
            parameter_name_ids = model.encode(
                f'"{parameter_name}"'
            )[0].tolist()

            if parameter_generated_ids == parameter_name_ids:
                return (
                    State.PARAMETER_COLON,
                    parameter_name,
                    len(generated_ids),
                )

    return (
        State.PARAMETER_NAME,
        None,
        len(generated_ids),
    )
