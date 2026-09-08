from json import load, JSONDecodeError
from pydantic import BaseModel, ValidationError


class PromptData(BaseModel):
    prompt: str


class ParameterInfo(BaseModel):
    type: str


class FunctionFormat(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterInfo]
    returns: ParameterInfo


def open_file(prompt_file: str) -> list:
    """Load JSON data from a file.

    Args:
        prompt_file: Path to the JSON file

    Returns:
        The parsed JSON data as a file.
        Returns an empty list if the file is not found
        or the JSON in invlid.

    """

    try:
        with open(prompt_file, 'r', encoding='UTF-8') as json_file:
            return load(json_file)
    except FileNotFoundError:
        print("not found file")
        return []
    except JSONDecodeError:
        print("It is not the correct JSON format")
        return []


def check_prompts(prompts: list) -> list[str]:
    """Validate prompt data and extract prompt strings.

    Args:
        prompts: Raw prompt data loaded from JSON.

    Returns:
        A list containing all valid prompt strings.

    """

    prompt_list: list[str] = []
    for item in prompts:
        try:
            prompt_data = PromptData.model_validate(item)
            prompt_text = prompt_data.prompt
            prompt_list.append(prompt_text)
        except ValidationError as e:
            print(f"Invalid prompt data: {e}")
    return prompt_list


def check_functions(functions: list) -> list[FunctionFormat]:
    """Validate raw function definitions.

    Args:

        functions: Raw function definition data loaded from JSON.

    Returns:

        A list of validated FunctionFormat objects.

    """
    functions_list: list[FunctionFormat] = []
    for item in functions:
        try:
            function_data = FunctionFormat.model_validate(item)
            functions_list.append(function_data)
        except ValidationError as e:
            print(f"Invalid function data: {e}")
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
        "User prompt"
        f"{prompt}"
    )
