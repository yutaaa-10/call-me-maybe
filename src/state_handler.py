from llm_sdk.llm_sdk import Small_LLM_Model
from .models import State, FunctionFormat


def handle_function_name(
    model: Small_LLM_Model,
    function_generated_ids: list[int],
    functions_list: list[FunctionFormat],
    generated_ids: list[int],
    next_token_id: int,
) -> tuple[State, FunctionFormat | None, int]:
    """Handle state transitions while generating a function name.
    Append the newly generated token and compare the generated sequence
    with the available function names. When a complete function name is
    matched, select the function and move to the separator state.

    Args:
        model: Language model used to encode function names.
        function_generated_ids: Function-name tokens generated so far.
        functions_list: Available function definitions.
        generated_ids: Token IDs generated for the complete output.
        next_token_id: Token ID generated in the current step.

    Returns:
        The next state, selected function if matched, and state position.

    """

    function_generated_ids.append(next_token_id)
    for function in functions_list:
        function_name_ids = model.encode(f'"{function.name}"')[0].tolist()
        if function_generated_ids == function_name_ids:
            selected_function = function
            state = State.FUNCTION_SEPARATOR
            state_start_position = len(generated_ids)

            return (
                state,
                selected_function,
                state_start_position
            )

    return (
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
    """Handle state transitions while generating a parameter name.
    Append the newly generated token and compare the generated sequence
    with the parameter names of the selected function. When a complete
    parameter name is matched, move to the parameter colon state.

    Args:
        model: Language model used to encode parameter names.
        parameter_generated_ids: Parameter-name tokens generated so far.
        selected_function: Function selected for the current call.
        generated_ids: Token IDs generated for the complete output.
        next_token_id: Token ID generated in the current step.

    Returns:
        The next state, selected parameter if matched, and state position.

    """
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


def string_is_complete(
    value_generated_ids: list[int],
    selected_function: FunctionFormat | None,
    selected_parameter: str,
    model: Small_LLM_Model,
) -> bool:
    """Check whether the generated parameter value is complete.
    Decode the generated value tokens and validate the result according
    to the type of the selected parameter. Numbers must be convertible
    to float, while strings must have valid opening and closing quotes.

    Args:
        value_generated_ids: Parameter-value tokens generated so far.
        selected_function: Function selected for the current call.
        selected_parameter: Name of the parameter currently generated.
        model: Language model used to decode generated tokens.

    Returns:
        True if the parameter value is complete, otherwise False.

    """

    if not value_generated_ids:
        return False
    if selected_function is None:
        return False

    # 今できているvalueをテキストに戻す
    value_text = model.decode(value_generated_ids)

    # stringならば、"a"などの最低3文字以上か、どうかなどを判定
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
    """Handle state transitions while generating a parameter value.
    Process the newly generated token according to the parameter type.
    For numbers, detect separators or the end of the parameters object.
    For other supported types, continue generation until the value is
    complete. Completed parameters are recorded before changing state.

    Args:
        model: Language model used to decode generated tokens.
        selected_function: Function selected for the current call.
        selected_parameter: Parameter currently being generated.
        next_token_id: Token ID generated in the current step.
        generated_ids: Token IDs generated for the complete output.
        parameter_generated_ids: Parameter-name tokens generated so far.
        value_generated_ids: Parameter-value tokens generated so far.
        completed_parameters: Names of parameters already completed.

    Returns:
        The next state, selected parameter, updated parameter-name tokens,
        updated value tokens, and state position.]

    """

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
        # valueの後に,が来たら
        if next_token_text.startswith(","):
            completed_parameters.append(selected_parameter)

            parameter_generated_ids = []
            value_generated_ids = []
            selected_parameter = None

            # 次のparameterの開始「"」まで生成されている場合、そのToken IDを保存する
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

        # valueの後に}が来たら
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

        return (
            State.PARAMETER_VALUE,
            selected_parameter,
            parameter_generated_ids,
            value_generated_ids,
            len(generated_ids),
        )

    value_generated_ids.append(next_token_id)

    if string_is_complete(
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

        return (
            state,
            selected_parameter,
            parameter_generated_ids,
            value_generated_ids,
            len(generated_ids),
        )
    return (
        State.PARAMETER_VALUE,
        selected_parameter,
        parameter_generated_ids,
        value_generated_ids,
        len(generated_ids),
    )
