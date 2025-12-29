
"""
Configuration management for YASS-RAG.
"""
import os
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Supported file types for indexing
SUPPORTED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.txt', '.md', '.html', '.htm',
    '.json', '.csv', '.xlsx', '.xls', '.pptx',
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs',
    '.rb', '.php', '.swift', '.kt', '.scala', '.r', '.sql'
}

# MIME type mappings for Google Docs export
GOOGLE_DOCS_EXPORT_MIMES = {
    'application/vnd.google-apps.document': ('application/pdf', '.pdf'),
    'application/vnd.google-apps.spreadsheet': ('text/csv', '.csv'),
    'application/vnd.google-apps.presentation': ('application/pdf', '.pdf'),
}

DEFAULT_MODEL = "gemini-2.5-flash"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 60
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class RAGConfig:
    """Global RAG configuration for the entire MCP server.

    Manages all settings for:
    - API credentials (Gemini, Google Drive)
    - Model and generation parameters
    - Polling and async behavior
    - Drive sync settings
    - Project/store defaults
    - Retrieval and response settings
    """

    def __init__(self):
        self.reset_to_defaults()

    def reset_to_defaults(self):
        """Reset all settings to defaults."""

        # === API Configuration ===
        self.gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")
        self.google_credentials_path: str | None = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.google_oauth_path: str | None = os.environ.get("GOOGLE_OAUTH_CREDENTIALS")

        # === Model Settings ===
        self.model: str = DEFAULT_MODEL
        self.temperature: float = 0.7
        self.max_output_tokens: int = 2048
        self.top_p: float = 0.95
        self.top_k: int = 40

        # === Polling & Async Settings ===
        self.poll_interval_seconds: int = POLL_INTERVAL_SECONDS
        self.max_poll_attempts: int = MAX_POLL_ATTEMPTS
        self.async_uploads: bool = False  # If True, don't wait for upload completion
        self.batch_size: int = 10  # Files per batch in async mode
        self.concurrent_uploads: int = 3  # Max concurrent uploads

        # === Google Drive Settings ===
        self.default_drive_folder: str | None = None  # Default folder URL/ID
        self.drive_recursive: bool = True
        self.drive_max_files: int = 100
        self.drive_file_extensions: list[str] | None = None  # None = all supported
        self.auto_sync_enabled: bool = False
        self.sync_interval_minutes: int = 60

        # === Project/Store Settings ===
        self.project_store: str | None = None  # Default store for project files
        self.default_stores: list[str] = []  # Stores to search by default
        self.auto_create_store: bool = True  # Create store if doesn't exist

        # === Retrieval Settings ===
        self.max_chunks: int = 10
        self.min_relevance_score: float = 0.0
        self.include_metadata: bool = True
        self.chunk_overlap_context: bool = True

        # === Response Settings ===
        self.include_citations: bool = True
        self.citation_style: str = "inline"  # "inline", "footnote", "end"
        self.response_format: str = "markdown"  # "markdown", "json"
        self.system_prompt: str = ""

        # === File Filtering ===
        self.supported_extensions: set = SUPPORTED_EXTENSIONS.copy()
        self.max_file_size_mb: int = 100
        self.skip_hidden_files: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (hides sensitive values)."""
        return {
            # API (masked)
            "gemini_api_key": "***" if self.gemini_api_key else None,
            "google_credentials_path": self.google_credentials_path,
            "google_oauth_path": self.google_oauth_path,

            # Model
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,

            # Polling & Async
            "poll_interval_seconds": self.poll_interval_seconds,
            "max_poll_attempts": self.max_poll_attempts,
            "async_uploads": self.async_uploads,
            "batch_size": self.batch_size,
            "concurrent_uploads": self.concurrent_uploads,

            # Drive
            "default_drive_folder": self.default_drive_folder,
            "drive_recursive": self.drive_recursive,
            "drive_max_files": self.drive_max_files,
            "drive_file_extensions": self.drive_file_extensions,
            "auto_sync_enabled": self.auto_sync_enabled,
            "sync_interval_minutes": self.sync_interval_minutes,

            # Project/Store
            "project_store": self.project_store,
            "default_stores": self.default_stores,
            "auto_create_store": self.auto_create_store,

            # Retrieval
            "max_chunks": self.max_chunks,
            "min_relevance_score": self.min_relevance_score,
            "include_metadata": self.include_metadata,
            "chunk_overlap_context": self.chunk_overlap_context,

            # Response
            "include_citations": self.include_citations,
            "citation_style": self.citation_style,
            "response_format": self.response_format,
            "system_prompt": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt,

            # Files
            "supported_extensions": list(self.supported_extensions),
            "max_file_size_mb": self.max_file_size_mb,
            "skip_hidden_files": self.skip_hidden_files,
        }

    def from_dict(self, config: dict[str, Any]):
        """Update config from dictionary."""
        for key, value in config.items():
            if hasattr(self, key):
                # Special handling for sets
                if key == "supported_extensions" and isinstance(value, list):
                    setattr(self, key, set(value))
                else:
                    setattr(self, key, value)

    def get_effective_api_key(self) -> str:
        """Get the effective Gemini API key."""
        key = self.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not configured. Use configure_rag or set environment variable.")
        return key


# Global RAG configuration instance
rag_config = RAGConfig()
