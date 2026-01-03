"""
Utility functions for YASS-RAG.
"""

import asyncio
import fcntl
import json
import os
import re
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from tqdm import tqdm

try:
    import msvcrt  # Windows
except ImportError:
    msvcrt = None


def _handle_error(e: Exception) -> str:
    """Consistent error formatting with context.

    Args:
        e: The exception to format

    Returns:
        Formatted error message with context and suggested actions
    """
    error_type = type(e).__name__
    error_msg = str(e)

    # HTTP errors
    if "404" in error_msg or "not found" in error_msg.lower():
        return f"## Error: Resource Not Found\n\n{error_msg}\n\n**Check:** Verify that resource ID or name is correct."
    elif "403" in error_msg or "permission" in error_msg.lower():
        return f"## Error: Access Denied\n\n{error_msg}\n\n**Check:** Verify your API credentials and permissions."
    elif "429" in error_msg or "rate limit" in error_msg.lower():
        return f"## Error: Rate Limit Exceeded\n\n{error_msg}\n\n**Check:** Reduce request frequency or wait before retrying."
    elif "500" in error_msg or "internal" in error_msg.lower():
        return f"## Error: Service Error\n\n{error_msg}\n\n**Action:** This is a temporary issue. Please try again later."

    # API key errors
    if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
        return f"## Error: Invalid API Key\n\n{error_msg}\n\n**Action:** Check your GEMINI_API_KEY configuration."

    # Timeout errors
    if isinstance(e, TimeoutError):
        return "## Error: Operation Timed Out\n\nThe operation did not complete within the expected time.\n\n**Action:** Try again with a larger timeout or reduce the operation scope."

    # Generic error
    return f"## Error ({error_type})\n\n{error_msg}"


# Input Validation
def validate_store_name(store_name: str) -> str:
    """Validate store name format.

    Args:
        store_name: Store name to validate

    Returns:
        Validated store name

    Raises:
        ValueError: If store name format is invalid
    """
    if not store_name.startswith("fileSearchStores/"):
        raise ValueError(f"Invalid store name '{store_name}'. Must start with 'fileSearchStores/'")
    return store_name


def sanitize_system_prompt(prompt: str) -> str:
    """Basic sanitization to prevent prompt injection.

    Args:
        prompt: System prompt to sanitize

    Returns:
        Sanitized prompt string
    """
    if not prompt:
        return prompt

    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r"<script.*?>.*?</script>",
        r"on\w+\s*=",  # onclick=, onerror=, etc.
    ]
    for pattern in dangerous_patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE)
    return prompt.strip()


def validate_drive_url(url: str) -> str:
    """Validate Google Drive URL format.

    Args:
        url: Drive URL or folder ID

    Returns:
        Validated URL or folder ID

    Raises:
        ValueError: If URL format is invalid
    """
    url = url.strip()

    # Allow just folder ID
    if re.match(r"^[a-zA-Z0-9_-]+$", url):
        return url

    # Validate URL format
    if "drive.google.com" not in url:
        raise ValueError("Invalid Google Drive URL")

    # Check for valid patterns
    valid_patterns = [
        r"https?://drive\.google\.com/drive/folders/[a-zA-Z0-9_-]+",
        r"https?://drive\.google\.com/drive/u/\d+/folders/[a-zA-Z0-9_-]+",
    ]

    if not any(re.match(p, url) for p in valid_patterns):
        raise ValueError("Invalid Google Drive folder URL format")

    return url


# Rate Limiting
class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, rate: float, per: float = 60.0):
        """
        Args:
            rate: Number of requests allowed
            per: Time window in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = threading.Lock()

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for rate limiting a function."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_check
                self.last_check = now
                self.allowance += elapsed * (self.rate / self.per)

                if self.allowance > self.rate:
                    self.allowance = self.rate

                if self.allowance < 1.0:
                    sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                    time.sleep(sleep_time)
                    self.allowance = 0.0
                    self.last_check = time.time()  # Update after sleep
                else:
                    self.allowance -= 1.0

            return func(*args, **kwargs)

        return wrapper


# Retry Logic
def retry_async(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Async retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for backoff delay
        exceptions: Tuple of exception types to retry on
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    delay = backoff_factor ** (attempt + 1)
                    await asyncio.sleep(delay)
            raise RuntimeError(f"Retry failed after {max_attempts} attempts") from last_exception

        return wrapper

    return decorator


# File Locking
@contextmanager
def file_lock(file: Any):
    """Context manager for file-based locking (Unix)."""
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock_windows(file: Any):
    """Context manager for file-based locking (Windows)."""
    if msvcrt is None:
        raise RuntimeError("msvcrt not available on this platform")
    try:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        yield
    finally:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]


# Use appropriate file lock based on OS
file_context = file_lock_windows if msvcrt else file_lock


# Progress Tracking
def track_progress(items: list[Any], description: str = "Processing") -> Generator[Any, None, None]:
    """Track progress with tqdm, but only if not in MCP context.

    Args:
        items: Items to iterate over
        description: Progress bar description

    Yields:
        Each item from the list
    """
    import sys

    # Check if we're in MCP mode (stdout is for communication)
    if hasattr(sys, "_mcp_mode") or not os.isatty(1):
        yield from items
    else:
        # tqdm doesn't support yield from, keeping for loop for progress bar
        for item in tqdm(items, desc=description):  # noqa: UP028
            yield item


# Batch Processing
async def process_in_batches(
    items: list[Any],
    batch_size: int,
    processor: Callable[..., Any],
    max_workers: int = 3,
    description: str = "Processing",
) -> tuple[list[Any], list[tuple[Any, Exception]]]:
    """Process items in parallel batches.

    Args:
        items: Items to process
        batch_size: Number of items per batch
        processor: Async function to process each item
        max_workers: Maximum concurrent workers
        description: Progress bar description

    Returns:
        Tuple of (results, errors)
    """
    results = []
    errors = []

    async def process_batch(batch: list[Any]) -> tuple[list[Any], list[tuple[Any, Exception]]]:
        batch_results = []
        batch_errors = []
        for item in batch:
            try:
                result = await processor(item)
                batch_results.append((item, result))
            except Exception as e:
                batch_errors.append((item, e))
        return batch_results, batch_errors

    # Split into batches
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    # Process batches
    for batch in track_progress(batches, description):
        batch_results, batch_errors = await process_batch(batch)
        results.extend(batch_results)
        errors.extend(batch_errors)

    return results, errors


# Response Formatting
class ResponseFormatter:
    """Utility for formatting MCP tool responses."""

    @staticmethod
    def json(success: bool, **data: Any) -> str:
        """Format response as JSON.

        Args:
            success: Whether operation was successful
            **data: Additional data to include

        Returns:
            JSON formatted string
        """
        return json.dumps({"success": success, **data}, indent=2)

    @staticmethod
    def markdown(title: str, content: str | list[str] = "") -> str:
        """Format response as Markdown.

        Args:
            title: Section title
            content: Content string or list of lines

        Returns:
            Markdown formatted string
        """
        lines = [f"## {title}"]
        if isinstance(content, str):
            if content:
                lines.append(content)
        else:
            lines.extend(content)
        return "\n".join(lines)

    @staticmethod
    def error(title: str, message: str, hint: str | None = None) -> str:
        """Format error response.

        Args:
            title: Error title
            message: Error message
            hint: Optional hint for resolution

        Returns:
            Formatted error string
        """
        lines = [f"## Error: {title}", "", message]
        if hint:
            lines.append("", f"**{hint}**")
        return "\n".join(lines)

    @staticmethod
    def success(message: str, details: list[str] | None = None) -> str:
        """Format success response.

        Args:
            message: Success message
            details: Optional list of detail items

        Returns:
            Formatted success string
        """
        lines: list[str] = [f"✅ {message}"]
        if details:
            detail_lines: list[str] = [f"- {detail}" for detail in details]
            lines.extend([""])
            lines.extend(detail_lines)
        return "\n".join(lines)


# Tool Handler Decorator
def tool_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for consistent error handling in tool functions.

    Args:
        func: Tool function to wrap

    Returns:
        Wrapped function with error handling
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return _handle_error(e)

    return wrapper
