
"""
Gemini API service helpers.
"""
import asyncio
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from google import genai
except ImportError as e:
    raise ImportError("Please install google-genai: uv add google-genai (or uv sync)") from e

from ..config import rag_config
from ..logging import get_logger

logger = get_logger("gemini")


# Retry decorator for API calls - retries on common transient errors
def _create_retry_decorator(max_attempts: int = 3):
    """Create a retry decorator with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying API call (attempt {retry_state.attempt_number}/{max_attempts}): "
            f"{retry_state.outcome.exception() if retry_state.outcome else 'unknown error'}"
        ),
    )


@_create_retry_decorator(max_attempts=3)
def _get_gemini_client() -> genai.Client:
    """Get or create Gemini client using configured API key."""
    api_key = rag_config.get_effective_api_key()
    logger.debug("Creating Gemini client")
    return genai.Client(api_key=api_key)


async def _wait_for_operation(
    client: genai.Client, operation: Any, max_attempts: int | None = None
) -> Any:
    """Wait for a long-running operation to complete using configured polling settings."""
    poll_interval = rag_config.poll_interval_seconds
    max_attempts = max_attempts or rag_config.max_poll_attempts

    attempts = 0
    logger.debug(f"Waiting for operation (max {max_attempts} attempts, {poll_interval}s interval)")

    while not operation.done and attempts < max_attempts:
        await asyncio.sleep(poll_interval)
        try:
            operation = client.operations.get(operation)
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Transient error polling operation (attempt {attempts + 1}): {e}")
            # Continue polling despite transient errors
        attempts += 1

    if not operation.done:
        logger.error(f"Operation timed out after {max_attempts * poll_interval}s")
        raise TimeoutError(f"Operation timed out after {max_attempts * poll_interval}s")

    logger.debug(f"Operation completed after {attempts} poll attempts")
    return operation


def _format_store_markdown(store: Any) -> str:
    """Format a store object as markdown."""
    lines = [
        f"### {getattr(store, 'display_name', 'Unnamed Store')}",
        f"- **Name**: `{store.name}`",
    ]
    if hasattr(store, 'create_time') and store.create_time:
        lines.append(f"- **Created**: {store.create_time}")
    return "\n".join(lines)


def _format_store_json(store: Any) -> dict[str, Any]:
    """Format a store object as JSON."""
    return {
        "name": store.name,
        "display_name": getattr(store, 'display_name', None),
        "create_time": str(getattr(store, 'create_time', None)),
    }


def _format_citations_markdown(grounding_metadata: Any) -> str:
    """Format citation metadata as markdown."""
    if not grounding_metadata:
        return ""

    lines = ["\n---\n### Citations"]

    if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
        for i, chunk in enumerate(grounding_metadata.grounding_chunks, 1):
            if hasattr(chunk, 'retrieved_context'):
                ctx = chunk.retrieved_context
                title = getattr(ctx, 'title', f'Source {i}')
                uri = getattr(ctx, 'uri', 'N/A')
                lines.append(f"\n**[{i}] {title}**")
                if uri and uri != 'N/A':
                    lines.append(f"- URI: `{uri}`")

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_citations_json(grounding_metadata: Any) -> dict[str, Any]:
    """Format citation metadata as JSON."""
    if not grounding_metadata:
        return {}

    result = {"chunks": [], "supports": []}

    if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
        for chunk in grounding_metadata.grounding_chunks:
            if hasattr(chunk, 'retrieved_context'):
                ctx = chunk.retrieved_context
                result["chunks"].append({
                    "title": getattr(ctx, 'title', None),
                    "uri": getattr(ctx, 'uri', None),
                })

    return result
