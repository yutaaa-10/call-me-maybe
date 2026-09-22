from json import load, JSONDecodeError
from pydantic import ValidationError
from .models import PromptData, FunctionFormat
import sys


def open_file(input_path: str) -> list[object] | None:
    """Load a JSON array from an input file.

    Args:
        input_path: Path to a UTF-8 JSON file.

    Returns:
        The loaded list, or None if reading or validation fails.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as json_file:
            data = load(json_file)
    except FileNotFoundError:
        print(
            f"Error: Input file not found: {input_path}",
            file=sys.stderr,
        )
        return None
    except JSONDecodeError as error:
        print(
            f"Error: Invalid JSON in {input_path} "
            f"(line {error.lineno}, column {error.colno}): "
            f"{error.msg}",
            file=sys.stderr,
        )
        return None
    except UnicodeDecodeError:
        print(
            f"Error: Input file is not valid UTF-8: {input_path}",
            file=sys.stderr,
        )
        return None
    except OSError as error:
        print(
            f"Error: Cannot read input file {input_path}: {error}",
            file=sys.stderr,
        )
        return None

    if not isinstance(data, list):
        print(
            f"Error: Expected a JSON array in {input_path}.",
            file=sys.stderr,
        )
        return None

    return data


def check_prompts(prompts: list[object]) -> list[str] | None:
    """Validate prompt data and extract prompt strings.

    Args:
        prompts: Raw prompt data loaded from JSON.

    Returns:
        A list containing all valid prompt strings.

    """
    if not prompts:
        print(
            "Error: The prompt list is empty.",
            file=sys.stderr,
        )
        return None

    prompt_list: list[str] = []

    for index, item in enumerate(prompts, start=1):
        try:
            prompt_data = PromptData.model_validate(item)
        except ValidationError as error:
            print(
                f"Error: Invalid prompt entry #{index}: {error}",
                file=sys.stderr,
            )
            return None

        prompt_list.append(prompt_data.prompt)

    return prompt_list


def check_functions(
    functions: list[object]
) -> list[FunctionFormat] | None:
    """Validate raw function definitions.

    Args:
        functions: Raw function definition data loaded from JSON.

    Returns:
        A list of validated FunctionFormat objects.

    """
    if not functions:
        print(
            "Error: The function definition list is empty.",
            file=sys.stderr,
        )
        return None

    functions_list: list[FunctionFormat] = []
    function_names: set[str] = set()

    for index, item in enumerate(functions, start=1):
        try:
            function_data = FunctionFormat.model_validate(item)
        except ValidationError as error:
            print(
                f"Error: Invalid function entry #{index}: {error}",
                file=sys.stderr,
            )
            return None

        if function_data.name in function_names:
            print(
                f"Error: Duplicate function name at entry #{index}: "
                f"{function_data.name}",
                file=sys.stderr,
            )
            return None

        function_names.add(function_data.name)
        functions_list.append(function_data)

    return functions_list


def build_functions_text(functions_list: list[FunctionFormat]) -> str:
    """Convert the verified functions into appropriate strings.

    Args:
        The verified functions.

    Returns:
        A formatted string describing the available functions,
        their parameters, and return types.

    """

    functions_text = ""
    for function in functions_list:
        functions_text += f"name: {function.name}\n"
        functions_text += f"description: {function.description}\n"
        for parameter_name, parameter_info in function.parameters.items():
            functions_text += f"parameter_name: {parameter_name}\n"
            functions_text += f"parameter_info: {parameter_info.type}\n"
        functions_text += f"returns: {function.returns.type}\n"
    return functions_text


def build_context(prompt: str, functions_text: str) -> str:
    """Build the input context for the language model.

    Args:
        prompt: User request to convert into a function call.
        functions_text: Text describing the available functions.

    Returns:
        A formatted context containing the functions and user request.

    """

    return (
        "Available functions:\n"
        f"{functions_text}\n"
        "Select the function whose description best matches "
        "the user's requested operation.\n"
        "Use the function definition to determine the parameters.\n"
        "\n"
        "Examples of repacement values:\n"
        'with asteriskd -> "*"\n'
        'with hyphens -> "-"\n'
        'with NUMBERS -> "NUMBERS"\n'
        "\n"
        "For regex parameters, generate"
        "the pattern that matches the text to be replaced.\n"
        "For replacement parameters, "
        "generate the new text that should replace it.\n"
        "\n"
        f"User request: {prompt}\n"
        "Function call:\n"
    )
