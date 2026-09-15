from llm_sdk.llm_sdk import Small_LLM_Model
from .models import State, FunctionFormat

def handle_function_name(
        model: Small_LLM_Model,
        function_generated_ids: list[int],
        functions_list: list[FunctionFormat],
        generated_ids: list[int],
        next_token_id: int,
    ) -> tuple[State, FunctionFormat | None, int]:

    function_generated_ids.append(next_token_id)
    for function in functions_list:
        function_name_ids = model.encode(f'"{function.name}"')[0].tolist()
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
        State.FUNCTION_NAME,
        None,
        len(generated_ids),
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

def value_is_complete(
        value_generated_ids: list[int],
        selected_function: FunctionFormat | None,
        selected_parameter: str,
        model: Small_LLM_Model,
    ) -> bool:

    if not value_generated_ids:
        return False

    if selected_function is None:
        return False

    parameter_info = selected_function.parameters[selected_parameter]
    parameter_type = parameter_info.type

    value_text = model.decode(value_generated_ids)

    if parameter_type == "number":
        try:
            float(value_text)
            return True
        except ValueError:
            return False
    if parameter_type == "string":
        if len(value_text) < 3:
            return False
        if not value_text.startswith('"'):
            return False
        if not value_text.endswith('"'):
            return False
        if value_text.count('"') != 2:
            return False
        return True



def handle_parameter_value(
        model: Small_LLM_Model,
        selected_function: FunctionFormat | None,
        selected_parameter: str | None,
        next_token_id: int,
        generated_ids: list[int],
        parameter_generated_ids: list[int],
        value_generated_ids: list[int],
        completed_parameters: list[str],
) -> tuple[State, str | None, list[int], list[int], int]:

    if selected_function is None or selected_parameter is None:
        return (
            State.PARAMETER_VALUE,
            selected_parameter,
            parameter_generated_ids,
            value_generated_ids,
            len(generated_ids),
        )

    parameter_type = selected_function.parameters[selected_parameter].type
    next_token_text = model.decode([next_token_id])

    if parameter_type == "number":
        if next_token_text.startswith(","):
            completed_parameters.append(selected_parameter)

            parameter_generated_ids = []
            value_generated_ids = []
            selected_parameter = None

            if next_token_text.startswith(',"'):
                parameter_generated_ids = (
                    model.encode('"')[0].tolist()
                )

            return (
                State.PARAMETER_NAME,
                selected_parameter,
                parameter_generated_ids,
                value_generated_ids,
                len(generated_ids),
            )

        if next_token_text.startswith("}"):
            completed_parameters.append(selected_parameter)

            value_generated_ids = []
            selected_parameter = None

            return (
                State.END,
                selected_parameter,
                parameter_generated_ids,
                value_generated_ids,
                len(generated_ids),
            )

        value_generated_ids.append(next_token_id)

        return(
            State.PARAMETER_VALUE,
            selected_parameter,
            parameter_generated_ids,
            value_generated_ids,
            len(generated_ids),
        )

    value_generated_ids.append(next_token_id)

    if value_is_complete(
        value_generated_ids,
        selected_function,
        selected_parameter,
        model,
    ):
        completed_parameters.append(selected_parameter)

        if len(completed_parameters) == len(selected_function.parameters):
            state = State.PARAMETERS_END
        else:
            state = State.PARAMETER_SEPARATOR

        return(
            state,
            selected_parameter,
            parameter_generated_ids,
            value_generated_ids,
            len(generated_ids),
        )
    return(
        State.PARAMETER_VALUE,
        selected_parameter,
        parameter_generated_ids,
        value_generated_ids,
        len(generated_ids),
    )






