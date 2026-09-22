import re


def is_complete_regex(text: str) -> bool:
    """Check whether the generated regex is syntactically valid."""

    if not text.startswith('"'):
        return False

    regex_text = text[1:]

    if regex_text == "":
        return False

    try:
        re.compile(regex_text)
    except re.error:
        return False

    return True


def is_complete_number(text: str) -> bool:
    """Check whether the given text represents a complete valid number.

    Args:
        text: Text to validate as a number.

    Returns:
        True if the text can be converted to a number, otherwise False.

    """

    try:
        float(text)
    except ValueError:
        return False

    return True


def is_valid_number_prefix(text: str) -> bool:
    """Check whether the text can still become a valid number.
    The text may be incomplete, such as an empty string or a minus sign,
    as long as additional characters could form a valid number.

    Args:
        text: Numeric text generated so far.

    Returns:
        True if the text is a valid numeric prefix, otherwise False.

    """

    if text == "":
        return True
    if text == "-":
        return True
    if text.count(".") > 1:
        return False
    start_index = 0
    if text[0] == "-":
        start_index = 1
    for character in text[start_index:]:
        if character != "." and not character.isdigit():
            return False
    return True
