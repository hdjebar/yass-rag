"""
Logging configuration for YASS-RAG.
"""
import logging
import os
import sys

# Create logger
logger = logging.getLogger("yass_rag")

# Default log level from environment or INFO
_log_level = os.environ.get("YASS_RAG_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _log_level, logging.INFO))

# Create console handler with formatting
_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.DEBUG)

# Create formatter
_formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler.setFormatter(_formatter)

# Add handler to logger (only if not already added)
if not logger.handlers:
    logger.addHandler(_handler)

# Prevent propagation to root logger
logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional name for child logger (e.g., 'gemini', 'drive')

    Returns:
        Logger instance
    """
    if name:
        return logger.getChild(name)
    return logger
