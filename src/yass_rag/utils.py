
"""
Utility functions for YASS-RAG.
"""
from pathlib import Path


def _validate_file_path(file_path: str, max_size_mb: int = 100) -> Path:
    """Validate a file path for security and existence.

    Args:
        file_path: The file path to validate
        max_size_mb: Maximum file size in MB

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is invalid, file doesn't exist, or exceeds size limit
    """
    # Resolve to absolute path
    path = Path(file_path).resolve()

    # Security: prevent path traversal attempts
    if ".." in file_path:
        raise ValueError("Path traversal patterns not allowed in file path")

    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Check file size
    file_size = path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise ValueError(
            f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({max_size_mb}MB)"
        )

    return path


def _handle_error(e: Exception) -> str:
    """Consistent error formatting."""
    error_type = type(e).__name__
    error_msg = str(e)

    if "404" in error_msg or "not found" in error_msg.lower():
        return f"Error: Resource not found. Details: {error_msg}"
    elif "403" in error_msg or "permission" in error_msg.lower():
        return f"Error: Permission denied. Details: {error_msg}"
    elif "429" in error_msg or "rate limit" in error_msg.lower():
        return f"Error: Rate limit exceeded. Details: {error_msg}"
    elif "api key" in error_msg.lower():
        return f"Error: Invalid API key. Details: {error_msg}"

    return f"Error ({error_type}): {error_msg}"
