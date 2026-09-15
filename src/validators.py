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

def is_valid_string_prefix(text: str) -> bool:
    """Check whether the text can still form a valid quoted string.
    The text must start with a double quote and may contain at most
    one opening and one closing quote.

    Args:
        text: String text generated so far.

    Returns:
        True if the text is a valid string prefix, otherwise False.

    """

    if text == "":
        return True
    if not text.startswith('"'):
        return False
    quote_count = text.count('"')
    if quote_count > 2:
        return False
    if quote_count == 2:
        if not text.endswith('"'):
            return False
    return True

