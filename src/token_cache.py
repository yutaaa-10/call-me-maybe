"""Cache repeated text encoding."""

from functools import lru_cache

from llm_sdk.llm_sdk import Small_LLM_Model


@lru_cache(maxsize=1024)
def _cached_encode(
    model: Small_LLM_Model,
    text: str,
) -> tuple[int, ...]:
    """Encode text and cache the resulting token IDs.

    Args:
        model: Language model used for token encoding.
        text: Text to encode into token IDs.

    Returns:
        The encoded token IDs as an immutable tuple.

    """

    return tuple(model.encode(text)[0].tolist())


def encode_ids(
    model: Small_LLM_Model,
    text: str,
) -> list[int]:
    """Return token IDs for the given text using the encoding cache.
    The cached immutable token sequence is converted to a new list before
    being returned.

    Args:
        model: Language model used for token encoding.
        text: Text to encode into token IDs.

    Returns:
        A new list containing the encoded token IDs.

    """

    return list(_cached_encode(model, text))
