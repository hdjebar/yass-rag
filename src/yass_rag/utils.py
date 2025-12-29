
"""
Utility functions for YASS-RAG.
"""
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
