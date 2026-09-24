"""Cache repeated text encoding."""

from functools import lru_cache

from llm_sdk.llm_sdk import Small_LLM_Model


@lru_cache(maxsize=1024)
def _cached_encode(
    model: Small_LLM_Model,
    text: str,
) -> tuple[int, ...]:
    """Cache encoded text as an immutable token sequence."""
    return tuple(model.encode(text)[0].tolist())


def encode_ids(
    model: Small_LLM_Model,
    text: str,
) -> list[int]:
    """Return a fresh list of cached token IDs."""
    return list(_cached_encode(model, text))
