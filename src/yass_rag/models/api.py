import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_MODEL = "gemini-2.5-flash"

# Valid patterns for Google Drive folder URLs
_DRIVE_URL_PATTERNS = [
    r"https?://drive\.google\.com/drive/folders/[a-zA-Z0-9_-]+",
    r"https?://drive\.google\.com/drive/u/\d+/folders/[a-zA-Z0-9_-]+",
]


def _validate_store_name(v: str) -> str:
    """Validate store name format."""
    if not v.startswith("fileSearchStores/"):
        raise ValueError(f"Invalid store name '{v}'. Must start with 'fileSearchStores/'")
    return v


def _validate_drive_folder_url(v: str) -> str:
    """Validate Google Drive folder URL or ID."""
    v = v.strip()

    # Allow just folder ID
    if re.match(r"^[a-zA-Z0-9_-]+$", v):
        return v

    # Validate URL format
    if "drive.google.com" not in v:
        raise ValueError("Invalid Google Drive URL")

    if not any(re.match(p, v) for p in _DRIVE_URL_PATTERNS):
        raise ValueError("Invalid Google Drive folder URL format")

    return v


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class CreateStoreInput(BaseModel):
    """Input for creating a new File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    display_name: str = Field(
        ...,
        description="Human-readable name for the File Search store",
        min_length=1,
        max_length=256,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'"
    )


class ListStoresInput(BaseModel):
    """Input for listing File Search stores."""

    model_config = ConfigDict(extra="forbid")

    limit: int | None = Field(default=20, ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class GetStoreInput(BaseModel):
    """Input for getting a specific File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_name: str = Field(..., min_length=1)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    _validate_store = field_validator("store_name")(_validate_store_name)


class DeleteStoreInput(BaseModel):
    """Input for deleting a File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_name: str = Field(..., min_length=1)

    _validate_store = field_validator("store_name")(_validate_store_name)


class UploadFileInput(BaseModel):
    """Input for uploading a file to a File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_name: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    display_name: str | None = Field(default=None, max_length=256)
    wait_for_completion: bool = Field(default=True)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    _validate_store = field_validator("store_name")(_validate_store_name)


class UploadTextInput(BaseModel):
    """Input for uploading text content directly."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_name: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1, max_length=256)
    wait_for_completion: bool = Field(default=True)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    _validate_store = field_validator("store_name")(_validate_store_name)


class SearchInput(BaseModel):
    """Input for searching files in a File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_names: list[str] = Field(..., min_length=1, max_length=10)
    query: str = Field(..., min_length=1)
    model: str | None = Field(default=DEFAULT_MODEL)
    include_citations: bool = Field(default=True)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    @field_validator("store_names")
    @classmethod
    def validate_store_formats(cls, v: list[str]) -> list[str]:
        """Validate all store name formats."""
        for name in v:
            _validate_store_name(name)
        return v


class ListFilesInput(BaseModel):
    """Input for listing files in a File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_name: str = Field(..., min_length=1)
    limit: int | None = Field(default=20, ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    _validate_store = field_validator("store_name")(_validate_store_name)


class DeleteFileInput(BaseModel):
    """Input for removing a file from a File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_name: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)

    _validate_store = field_validator("store_name")(_validate_store_name)


# Google Drive Sync Models
class SyncDriveFolderInput(BaseModel):
    """Input for syncing a Google Drive folder to a File Search store."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    store_name: str = Field(
        ...,
        description="Target File Search store resource name (e.g., 'fileSearchStores/abc123')",
        min_length=1,
    )
    folder: str = Field(
        ...,
        description="Google Drive folder URL or ID. Accepts full URL (e.g., 'https://drive.google.com/drive/folders/1ABC_xyz') or just the folder ID ('1ABC_xyz')",
        min_length=1,
    )
    recursive: bool = Field(default=True, description="Whether to include files from subfolders")
    file_extensions: list[str] | None = Field(
        default=None,
        description="Filter by file extensions (e.g., ['.pdf', '.docx']). If None, uses default supported types.",
    )
    max_files: int | None = Field(
        default=100,
        description="Maximum number of files to sync",
        ge=1,
        le=500,  # Reduced from 1000 for quota safety
    )
    credentials_path: str | None = Field(
        default=None,
        description="Path to OAuth credentials JSON or service account key. Uses env vars if not provided.",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    _validate_store = field_validator("store_name")(_validate_store_name)
    _validate_folder = field_validator("folder")(_validate_drive_folder_url)


class ListDriveFilesInput(BaseModel):
    """Input for listing files in a Google Drive folder."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    folder: str = Field(
        ...,
        description="Google Drive folder URL or ID. Accepts full URL (e.g., 'https://drive.google.com/drive/folders/1ABC_xyz') or just the folder ID",
        min_length=1,
    )
    recursive: bool = Field(default=False)
    max_files: int | None = Field(default=50, ge=1, le=500)
    credentials_path: str | None = Field(default=None)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    _validate_folder = field_validator("folder")(_validate_drive_folder_url)


# RAG Configuration Models
class CitationStyle(str, Enum):
    """Citation style for RAG responses."""

    INLINE = "inline"
    FOOTNOTE = "footnote"
    END = "end"


class ConfigureRAGInput(BaseModel):
    """Comprehensive RAG configuration input.

    Configure all aspects of the RAG pipeline: API keys, model settings,
    polling behavior, Drive sync, and response formatting.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # === API Configuration ===
    gemini_api_key: str | None = Field(
        default=None, description="Gemini API key. Get from https://aistudio.google.com/apikey"
    )
    google_credentials_path: str | None = Field(
        default=None, description="Path to Google service account JSON or OAuth credentials file"
    )

    # === Model Settings ===
    model: str | None = Field(
        default=None,
        description="Gemini model (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash')",
    )
    temperature: float | None = Field(
        default=None,
        description="Generation temperature (0.0-2.0). Lower = focused, higher = creative",
        ge=0.0,
        le=2.0,
    )
    max_output_tokens: int | None = Field(
        default=None, description="Maximum tokens in response", ge=1, le=8192
    )
    top_p: float | None = Field(
        default=None, description="Top-p (nucleus) sampling parameter", ge=0.0, le=1.0
    )
    top_k: int | None = Field(default=None, description="Top-k sampling parameter", ge=1, le=100)

    # === Polling & Async Settings ===
    poll_interval_seconds: int | None = Field(
        default=None, description="Seconds between polling for operation completion", ge=1, le=60
    )
    max_poll_attempts: int | None = Field(
        default=None, description="Maximum polling attempts before timeout", ge=1, le=600
    )
    async_uploads: bool | None = Field(
        default=None, description="If true, uploads return immediately without waiting for indexing"
    )
    batch_size: int | None = Field(
        default=None, description="Number of files per batch in async uploads", ge=1, le=50
    )
    concurrent_uploads: int | None = Field(
        default=None, description="Maximum concurrent upload operations", ge=1, le=10
    )

    # === Google Drive Settings ===
    default_drive_folder: str | None = Field(
        default=None, description="Default Google Drive folder URL or ID for sync operations"
    )
    drive_recursive: bool | None = Field(
        default=None, description="Whether to recursively sync subfolders"
    )
    drive_max_files: int | None = Field(
        default=None, description="Maximum files to sync from Drive", ge=1, le=1000
    )
    drive_file_extensions: list[str] | None = Field(
        default=None,
        description="File extensions to sync (e.g., ['.pdf', '.docx']). None = all supported",
    )
    auto_sync_enabled: bool | None = Field(
        default=None, description="Enable automatic periodic sync from Drive"
    )
    sync_interval_minutes: int | None = Field(
        default=None, description="Minutes between auto-sync operations", ge=5, le=1440
    )

    # === Project/Store Settings ===
    project_store: str | None = Field(
        default=None,
        description="Default File Search store for project files (e.g., 'fileSearchStores/abc123')",
    )
    default_stores: list[str] | None = Field(
        default=None, description="Stores to search by default when none specified"
    )
    auto_create_store: bool | None = Field(
        default=None, description="Automatically create store if it doesn't exist"
    )

    # === Retrieval Settings ===
    max_chunks: int | None = Field(
        default=None, description="Maximum document chunks to retrieve", ge=1, le=50
    )
    min_relevance_score: float | None = Field(
        default=None, description="Minimum relevance threshold (0.0-1.0)", ge=0.0, le=1.0
    )
    include_metadata: bool | None = Field(
        default=None, description="Include file metadata in context"
    )
    chunk_overlap_context: bool | None = Field(
        default=None, description="Include overlapping context from adjacent chunks"
    )

    # === Response Settings ===
    include_citations: bool | None = Field(
        default=None, description="Include source citations in responses"
    )
    citation_style: CitationStyle | None = Field(
        default=None, description="Citation style: 'inline', 'footnote', 'end'"
    )
    response_format: ResponseFormat | None = Field(
        default=None, description="Default output format: 'markdown' or 'json'"
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt for RAG queries (persona, instructions, domain context)",
        max_length=4000,
    )

    @field_validator("system_prompt")
    @classmethod
    def sanitize_prompt(cls, v: str | None) -> str | None:
        """Sanitize system prompt to prevent injection."""
        if not v:
            return v

        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r"<script.*?>.*?</script>",
            r"on\w+\s*=",  # onclick=, onerror=, etc.
        ]
        for pattern in dangerous_patterns:
            v = re.sub(pattern, "", v, flags=re.IGNORECASE)

        return v.strip() if v else v

    # === File Filtering ===
    max_file_size_mb: int | None = Field(
        default=None, description="Maximum file size in MB to process", ge=1, le=100
    )
    skip_hidden_files: bool | None = Field(
        default=None, description="Skip hidden files (starting with .)"
    )


class GetRAGConfigInput(BaseModel):
    """Input for getting RAG configuration."""

    model_config = ConfigDict(extra="forbid")

    show_sensitive: bool = Field(
        default=False, description="Show sensitive values like API keys (masked by default)"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ResetRAGConfigInput(BaseModel):
    """Input for resetting RAG configuration to defaults."""

    model_config = ConfigDict(extra="forbid")

    confirm: bool = Field(default=False, description="Set to true to confirm reset")
    preserve_api_keys: bool = Field(
        default=True, description="Keep API keys when resetting other settings"
    )
