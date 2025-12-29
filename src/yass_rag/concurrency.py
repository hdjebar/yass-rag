"""
Concurrency utilities for YASS-RAG.
"""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .config import rag_config
from .logging import get_logger

logger = get_logger("concurrency")

# Global semaphore for concurrent uploads (lazy initialized)
_upload_semaphore: asyncio.Semaphore | None = None


def _get_upload_semaphore() -> asyncio.Semaphore:
    """Get or create the upload semaphore based on config.

    Returns:
        Semaphore limiting concurrent uploads
    """
    global _upload_semaphore

    if _upload_semaphore is None:
        limit = rag_config.concurrent_uploads
        logger.debug(f"Initializing upload semaphore with limit={limit}")
        _upload_semaphore = asyncio.Semaphore(limit)

    return _upload_semaphore


def reset_upload_semaphore() -> None:
    """Reset the upload semaphore (useful when config changes)."""
    global _upload_semaphore
    _upload_semaphore = None
    logger.debug("Upload semaphore reset")


@asynccontextmanager
async def upload_slot() -> AsyncIterator[None]:
    """Context manager to acquire a slot for concurrent uploads.

    Usage:
        async with upload_slot():
            await do_upload()

    This limits concurrent uploads based on rag_config.concurrent_uploads.
    """
    semaphore = _get_upload_semaphore()
    logger.debug("Waiting for upload slot...")

    async with semaphore:
        logger.debug("Upload slot acquired")
        try:
            yield
        finally:
            logger.debug("Upload slot released")


async def batch_upload(
    items: list,
    upload_func,
    batch_size: int | None = None,
) -> list:
    """Execute uploads in batches with concurrency control.

    Args:
        items: List of items to upload
        upload_func: Async function to call for each item
        batch_size: Number of items per batch (defaults to config.batch_size)

    Returns:
        List of results from upload_func calls
    """
    batch_size = batch_size or rag_config.batch_size
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch)} items)")

        # Execute batch with concurrency control
        batch_tasks = []
        for item in batch:
            async def _upload_with_slot(item=item):
                async with upload_slot():
                    return await upload_func(item)

            batch_tasks.append(_upload_with_slot())

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)

    return results
