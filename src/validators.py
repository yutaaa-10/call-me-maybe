def is_complete_number(text: str) -> bool:
    """Check whether text is a complete valid number."""

    try:
        float(text)
    except ValueError:
        return False

    return True

def is_valid_number_prefix(text: str) -> bool:
    """Check whether text can still become a valid number."""
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

