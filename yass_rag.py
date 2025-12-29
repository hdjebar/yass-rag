"""
YASS-RAG: Yet Another Simple & Smart RAG

An MCP server that exposes Google's Gemini File Search API capabilities,
including a connector to sync Google Drive folders to File Search stores.

Installation (using uv):
    uv sync                    # Core dependencies only
    uv sync --extra drive      # With Google Drive support
    uv sync --all-extras       # Everything including dev tools

Environment Variables:
    GEMINI_API_KEY: Your Gemini API key (supports simple API key auth)
    
    For Google Drive sync (OAuth 2.0 or Service Account required - API keys NOT supported):
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
        - Best for: servers, automation, shared folders
        - Setup: Share Drive folder with service account email
    GOOGLE_OAUTH_CREDENTIALS: Path to OAuth client secrets JSON  
        - Best for: personal Drive access, desktop apps
        - Setup: Requires one-time browser consent flow

Note: Google Drive API does NOT support API keys because it accesses private user data.
      You must use OAuth 2.0 or a Service Account.

Usage:
    uv run yass-rag
"""

import os
import io
import json
import time
import tempfile
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# Gemini API
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Please install google-genai: uv add google-genai (or uv sync)")

# Google Drive API (optional - for sync features)
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    DRIVE_API_AVAILABLE = True
except ImportError:
    DRIVE_API_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 60

# Google Drive scopes
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

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


# ============================================================================
# RAG Configuration
# ============================================================================

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
        self.gemini_api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        self.google_credentials_path: Optional[str] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.google_oauth_path: Optional[str] = os.environ.get("GOOGLE_OAUTH_CREDENTIALS")
        
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
        self.default_drive_folder: Optional[str] = None  # Default folder URL/ID
        self.drive_recursive: bool = True
        self.drive_max_files: int = 100
        self.drive_file_extensions: Optional[List[str]] = None  # None = all supported
        self.auto_sync_enabled: bool = False
        self.sync_interval_minutes: int = 60
        
        # === Project/Store Settings ===
        self.project_store: Optional[str] = None  # Default store for project files
        self.default_stores: List[str] = []  # Stores to search by default
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
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    def from_dict(self, config: Dict[str, Any]):
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


# ============================================================================
# Initialize MCP Server
# ============================================================================

mcp = FastMCP("yass-rag")


# ============================================================================
# Pydantic Input Models
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class CreateStoreInput(BaseModel):
    """Input for creating a new File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    display_name: str = Field(
        ...,
        description="Human-readable name for the File Search store",
        min_length=1,
        max_length=256
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'"
    )


class ListStoresInput(BaseModel):
    """Input for listing File Search stores."""
    model_config = ConfigDict(extra='forbid')
    
    limit: Optional[int] = Field(default=20, ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class GetStoreInput(BaseModel):
    """Input for getting a specific File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_name: str = Field(..., min_length=1)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class DeleteStoreInput(BaseModel):
    """Input for deleting a File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_name: str = Field(..., min_length=1)


class UploadFileInput(BaseModel):
    """Input for uploading a file to a File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_name: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    display_name: Optional[str] = Field(default=None, max_length=256)
    wait_for_completion: bool = Field(default=True)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class UploadTextInput(BaseModel):
    """Input for uploading text content directly."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_name: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1, max_length=256)
    wait_for_completion: bool = Field(default=True)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SearchInput(BaseModel):
    """Input for searching files in a File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_names: List[str] = Field(..., min_length=1, max_length=10)
    query: str = Field(..., min_length=1)
    model: Optional[str] = Field(default=DEFAULT_MODEL)
    include_citations: bool = Field(default=True)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ListFilesInput(BaseModel):
    """Input for listing files in a File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_name: str = Field(..., min_length=1)
    limit: Optional[int] = Field(default=20, ge=1, le=100)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class DeleteFileInput(BaseModel):
    """Input for removing a file from a File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_name: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)


# Google Drive Sync Models
class SyncDriveFolderInput(BaseModel):
    """Input for syncing a Google Drive folder to a File Search store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    store_name: str = Field(
        ...,
        description="Target File Search store resource name (e.g., 'fileSearchStores/abc123')",
        min_length=1
    )
    folder: str = Field(
        ...,
        description="Google Drive folder URL or ID. Accepts full URL (e.g., 'https://drive.google.com/drive/folders/1ABC_xyz') or just the folder ID ('1ABC_xyz')",
        min_length=1
    )
    recursive: bool = Field(
        default=True,
        description="Whether to include files from subfolders"
    )
    file_extensions: Optional[List[str]] = Field(
        default=None,
        description="Filter by file extensions (e.g., ['.pdf', '.docx']). If None, uses default supported types."
    )
    max_files: Optional[int] = Field(
        default=100,
        description="Maximum number of files to sync",
        ge=1,
        le=1000
    )
    credentials_path: Optional[str] = Field(
        default=None,
        description="Path to OAuth credentials JSON or service account key. Uses env vars if not provided."
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ListDriveFilesInput(BaseModel):
    """Input for listing files in a Google Drive folder."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    folder: str = Field(
        ...,
        description="Google Drive folder URL or ID. Accepts full URL (e.g., 'https://drive.google.com/drive/folders/1ABC_xyz') or just the folder ID",
        min_length=1
    )
    recursive: bool = Field(default=False)
    max_files: Optional[int] = Field(default=50, ge=1, le=500)
    credentials_path: Optional[str] = Field(default=None)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


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
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    # === API Configuration ===
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Gemini API key. Get from https://aistudio.google.com/apikey"
    )
    google_credentials_path: Optional[str] = Field(
        default=None,
        description="Path to Google service account JSON or OAuth credentials file"
    )
    
    # === Model Settings ===
    model: Optional[str] = Field(
        default=None,
        description="Gemini model (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash')"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Generation temperature (0.0-2.0). Lower = focused, higher = creative",
        ge=0.0,
        le=2.0
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in response",
        ge=1,
        le=8192
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p (nucleus) sampling parameter",
        ge=0.0,
        le=1.0
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling parameter",
        ge=1,
        le=100
    )
    
    # === Polling & Async Settings ===
    poll_interval_seconds: Optional[int] = Field(
        default=None,
        description="Seconds between polling for operation completion",
        ge=1,
        le=60
    )
    max_poll_attempts: Optional[int] = Field(
        default=None,
        description="Maximum polling attempts before timeout",
        ge=1,
        le=600
    )
    async_uploads: Optional[bool] = Field(
        default=None,
        description="If true, uploads return immediately without waiting for indexing"
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Number of files per batch in async uploads",
        ge=1,
        le=50
    )
    concurrent_uploads: Optional[int] = Field(
        default=None,
        description="Maximum concurrent upload operations",
        ge=1,
        le=10
    )
    
    # === Google Drive Settings ===
    default_drive_folder: Optional[str] = Field(
        default=None,
        description="Default Google Drive folder URL or ID for sync operations"
    )
    drive_recursive: Optional[bool] = Field(
        default=None,
        description="Whether to recursively sync subfolders"
    )
    drive_max_files: Optional[int] = Field(
        default=None,
        description="Maximum files to sync from Drive",
        ge=1,
        le=1000
    )
    drive_file_extensions: Optional[List[str]] = Field(
        default=None,
        description="File extensions to sync (e.g., ['.pdf', '.docx']). None = all supported"
    )
    auto_sync_enabled: Optional[bool] = Field(
        default=None,
        description="Enable automatic periodic sync from Drive"
    )
    sync_interval_minutes: Optional[int] = Field(
        default=None,
        description="Minutes between auto-sync operations",
        ge=5,
        le=1440
    )
    
    # === Project/Store Settings ===
    project_store: Optional[str] = Field(
        default=None,
        description="Default File Search store for project files (e.g., 'fileSearchStores/abc123')"
    )
    default_stores: Optional[List[str]] = Field(
        default=None,
        description="Stores to search by default when none specified"
    )
    auto_create_store: Optional[bool] = Field(
        default=None,
        description="Automatically create store if it doesn't exist"
    )
    
    # === Retrieval Settings ===
    max_chunks: Optional[int] = Field(
        default=None,
        description="Maximum document chunks to retrieve",
        ge=1,
        le=50
    )
    min_relevance_score: Optional[float] = Field(
        default=None,
        description="Minimum relevance threshold (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    include_metadata: Optional[bool] = Field(
        default=None,
        description="Include file metadata in context"
    )
    chunk_overlap_context: Optional[bool] = Field(
        default=None,
        description="Include overlapping context from adjacent chunks"
    )
    
    # === Response Settings ===
    include_citations: Optional[bool] = Field(
        default=None,
        description="Include source citations in responses"
    )
    citation_style: Optional[CitationStyle] = Field(
        default=None,
        description="Citation style: 'inline', 'footnote', 'end'"
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="Default output format: 'markdown' or 'json'"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for RAG queries (persona, instructions, domain context)",
        max_length=4000
    )
    
    # === File Filtering ===
    max_file_size_mb: Optional[int] = Field(
        default=None,
        description="Maximum file size in MB to process",
        ge=1,
        le=100
    )
    skip_hidden_files: Optional[bool] = Field(
        default=None,
        description="Skip hidden files (starting with .)"
    )


class GetRAGConfigInput(BaseModel):
    """Input for getting RAG configuration."""
    model_config = ConfigDict(extra='forbid')
    
    show_sensitive: bool = Field(
        default=False,
        description="Show sensitive values like API keys (masked by default)"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ResetRAGConfigInput(BaseModel):
    """Input for resetting RAG configuration to defaults."""
    model_config = ConfigDict(extra='forbid')
    
    confirm: bool = Field(
        default=False,
        description="Set to true to confirm reset"
    )
    preserve_api_keys: bool = Field(
        default=True,
        description="Keep API keys when resetting other settings"
    )


# ============================================================================
# Helper Functions - Gemini
# ============================================================================

def _get_gemini_client() -> "genai.Client":
    """Get or create Gemini client using configured API key.

    Returns:
        genai.Client: Configured Gemini API client

    Raises:
        ValueError: If no API key is configured
    """
    api_key = rag_config.get_effective_api_key()
    return genai.Client(api_key=api_key)


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


def _wait_for_operation(client: genai.Client, operation: Any, max_attempts: Optional[int] = None) -> Any:
    """Wait for a long-running operation to complete using configured polling settings."""
    poll_interval = rag_config.poll_interval_seconds
    max_attempts = max_attempts or rag_config.max_poll_attempts
    
    attempts = 0
    while not operation.done and attempts < max_attempts:
        time.sleep(poll_interval)
        operation = client.operations.get(operation)
        attempts += 1
    
    if not operation.done:
        raise TimeoutError(f"Operation timed out after {max_attempts * poll_interval}s")
    
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


def _format_store_json(store: Any) -> Dict[str, Any]:
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


def _format_citations_json(grounding_metadata: Any) -> Dict[str, Any]:
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


# ============================================================================
# Helper Functions - Google Drive
# ============================================================================

import re

def _parse_drive_folder_id(folder_input: str) -> str:
    """Extract folder ID from a Google Drive URL or return the ID if already provided.

    Supports URLs like:
    - https://drive.google.com/drive/folders/1ABC_xyz
    - https://drive.google.com/drive/u/0/folders/1ABC_xyz
    - https://drive.google.com/drive/u/2/folders/1ABC_xyz?usp=sharing
    - drive.google.com/drive/folders/1ABC_xyz
    - Just the folder ID: 1ABC_xyz

    Args:
        folder_input: Either a full Google Drive URL or a folder ID

    Returns:
        The extracted folder ID
    """
    folder_input = folder_input.strip()

    # If it looks like a URL (contains drive.google.com or starts with http)
    if 'drive.google.com' in folder_input or folder_input.startswith('http'):
        # Simple pattern: look for /folders/ followed by the ID
        folders_pattern = r'/folders/([a-zA-Z0-9_-]+)'
        match = re.search(folders_pattern, folder_input)
        if match:
            return match.group(1)

        # Fallback: get last path segment before query params
        path = folder_input.split('?')[0].split('#')[0].rstrip('/')
        segments = path.split('/')
        if segments:
            return segments[-1]

    # Assume it's already a folder ID (validate it looks like one)
    # Google Drive folder IDs are typically 28-44 characters of alphanumeric, underscores, hyphens
    if re.match(r'^[a-zA-Z0-9_-]+$', folder_input):
        return folder_input

    # Return as-is if nothing matches (let the API return a proper error)
    return folder_input


def _get_drive_credentials(credentials_path: Optional[str] = None) -> Any:
    """Get Google Drive API credentials.

    Args:
        credentials_path: Optional path to credentials JSON file

    Returns:
        Google credentials object (service account or OAuth)

    Raises:
        ImportError: If Google Drive API packages are not installed
        ValueError: If no valid credentials are found
        google.auth.exceptions.RefreshError: If token refresh fails
    """
    if not DRIVE_API_AVAILABLE:
        raise ImportError(
            "Google Drive API not available. Install with: "
            "uv sync --extra drive (or: uv add google-api-python-client google-auth google-auth-oauthlib)"
        )

    creds = None

    # Try credentials path first
    cred_file = credentials_path or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    oauth_file = credentials_path or os.environ.get('GOOGLE_OAUTH_CREDENTIALS')
    token_file = Path.home() / '.gemini_mcp_token.json'

    # Check for existing token
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), DRIVE_SCOPES)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Token file is corrupted, remove it and continue
            print(f"Warning: Corrupted token file removed: {e}")
            token_file.unlink(missing_ok=True)
            creds = None

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                # Token refresh failed, need to re-authenticate
                print(f"Warning: Token refresh failed ({e}), re-authenticating...")
                creds = None
                token_file.unlink(missing_ok=True)

        if not creds:
            if cred_file and os.path.exists(cred_file):
                # Try service account first
                try:
                    creds = service_account.Credentials.from_service_account_file(
                        cred_file, scopes=DRIVE_SCOPES
                    )
                except (ValueError, KeyError) as e:
                    # Not a service account file, try OAuth
                    print(f"Note: Not a service account file ({e}), trying OAuth flow...")
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(cred_file, DRIVE_SCOPES)
                        creds = flow.run_local_server(port=0)
                    except Exception as oauth_e:
                        raise ValueError(f"Failed to authenticate with {cred_file}: {oauth_e}")
            elif oauth_file and os.path.exists(oauth_file):
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(oauth_file, DRIVE_SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    raise ValueError(f"Failed to authenticate with OAuth file {oauth_file}: {e}")
            else:
                raise ValueError(
                    "No Google credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or "
                    "GOOGLE_OAUTH_CREDENTIALS environment variable, or provide credentials_path."
                )

        # Save token for future use (OAuth tokens only, not service accounts)
        if hasattr(creds, 'refresh_token') and creds.refresh_token:
            try:
                # Create token file with restricted permissions (600 - owner read/write only)
                token_file.touch(mode=0o600, exist_ok=True)
                with open(token_file, 'w') as f:
                    f.write(creds.to_json())
                # Ensure permissions are correct even if file existed
                os.chmod(token_file, 0o600)
            except OSError as e:
                # Non-fatal: warn but continue
                print(f"Warning: Could not save token file: {e}")

    return creds


def _get_drive_service(credentials_path: Optional[str] = None) -> Any:
    """Get Google Drive API service.

    Args:
        credentials_path: Optional path to credentials JSON file

    Returns:
        Google Drive API service resource
    """
    creds = _get_drive_credentials(credentials_path)
    return build('drive', 'v3', credentials=creds)


def _list_drive_files(
    service: Any,
    folder_id: str,
    recursive: bool = True,
    max_files: int = 100,
    extensions: Optional[set] = None
) -> List[Dict[str, Any]]:
    """List files in a Google Drive folder.

    Args:
        service: Google Drive API service resource
        folder_id: Google Drive folder ID
        recursive: Whether to include files from subfolders
        max_files: Maximum number of files to return
        extensions: Set of file extensions to filter by (e.g., {'.pdf', '.docx'})

    Returns:
        List of file metadata dicts with 'id', 'name', 'mimeType', 'size', 'modifiedTime'
    """
    extensions = extensions or SUPPORTED_EXTENSIONS
    files = []
    folders_to_process = [folder_id]
    
    while folders_to_process and len(files) < max_files:
        current_folder = folders_to_process.pop(0)
        
        query = f"'{current_folder}' in parents and trashed = false"
        page_token = None
        
        while len(files) < max_files:
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, size, modifiedTime)',
                pageToken=page_token,
                pageSize=min(100, max_files - len(files))
            ).execute()
            
            for file in response.get('files', []):
                mime_type = file.get('mimeType', '')
                
                # Handle folders
                if mime_type == 'application/vnd.google-apps.folder':
                    if recursive:
                        folders_to_process.append(file['id'])
                    continue
                
                # Handle Google Docs (exportable)
                if mime_type in GOOGLE_DOCS_EXPORT_MIMES:
                    files.append(file)
                    continue
                
                # Check extension for regular files
                name = file.get('name', '')
                ext = os.path.splitext(name)[1].lower()
                if ext in extensions:
                    files.append(file)
            
            page_token = response.get('nextPageToken')
            if not page_token:
                break
    
    return files[:max_files]


def _download_drive_file(service: Any, file_info: Dict[str, Any], temp_dir: str) -> Optional[str]:
    """Download a file from Google Drive to a temp directory.

    Args:
        service: Google Drive API service resource
        file_info: File metadata dict with 'id', 'name', 'mimeType'
        temp_dir: Directory to download file to

    Returns:
        Path to downloaded file, or None if download failed
    """
    file_id = file_info['id']
    file_name = file_info['name']
    mime_type = file_info.get('mimeType', '')
    
    try:
        # Handle Google Docs (need to export)
        if mime_type in GOOGLE_DOCS_EXPORT_MIMES:
            export_mime, ext = GOOGLE_DOCS_EXPORT_MIMES[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
            file_name = os.path.splitext(file_name)[0] + ext
        else:
            request = service.files().get_media(fileId=file_id)
        
        file_path = os.path.join(temp_dir, file_name)
        
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        
        return file_path
    
    except Exception as e:
        print(f"Warning: Failed to download {file_name}: {e}")
        return None


# ============================================================================
# MCP Tools - File Search Store Management
# ============================================================================

@mcp.tool(
    name="create_store",
    annotations={
        "title": "Create File Search Store",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def create_store(params: CreateStoreInput) -> str:
    """Create a new Gemini File Search store for indexing documents."""
    try:
        client = _get_gemini_client()
        store = client.file_search_stores.create(config={'display_name': params.display_name})
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"success": True, "store": _format_store_json(store)}, indent=2)
        
        return f"""## File Search Store Created

{_format_store_markdown(store)}

**Next Steps:**
1. Upload files using `upload_file` or sync from Drive using `sync_drive_folder`
2. Search with `search`
"""
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="list_stores",
    annotations={
        "title": "List File Search Stores",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def list_stores(params: ListStoresInput) -> str:
    """List all File Search stores."""
    try:
        client = _get_gemini_client()
        stores = list(client.file_search_stores.list(config={'page_size': params.limit}))
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "success": True,
                "count": len(stores),
                "stores": [_format_store_json(s) for s in stores]
            }, indent=2)
        
        if not stores:
            return "## File Search Stores\n\nNo stores found. Create one using `create_store`."
        
        lines = [f"## File Search Stores ({len(stores)} found)\n"]
        for store in stores:
            lines.append(_format_store_markdown(store))
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="get_store",
    annotations={
        "title": "Get Store Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_store(params: GetStoreInput) -> str:
    """Get details of a specific File Search store."""
    try:
        client = _get_gemini_client()
        store = client.file_search_stores.get(name=params.store_name)
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"success": True, "store": _format_store_json(store)}, indent=2)
        
        return f"## File Search Store Details\n\n{_format_store_markdown(store)}"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="delete_store",
    annotations={
        "title": "Delete File Search Store",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def delete_store(params: DeleteStoreInput) -> str:
    """Delete a File Search store and all its indexed documents."""
    try:
        client = _get_gemini_client()
        client.file_search_stores.delete(name=params.store_name)
        return f"## Store Deleted\n\n`{params.store_name}` has been permanently deleted."
    except Exception as e:
        return _handle_error(e)


# ============================================================================
# MCP Tools - File Upload
# ============================================================================

@mcp.tool(
    name="upload_file",
    annotations={
        "title": "Upload File to Store",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def upload_file(params: UploadFileInput) -> str:
    """Upload and index a local file to a File Search store."""
    try:
        client = _get_gemini_client()

        # Validate file path
        file_path = Path(params.file_path).resolve()

        # Security: Prevent path traversal attacks
        # Ensure the resolved path doesn't escape to sensitive system directories
        sensitive_dirs = [
            Path('/etc'),
            Path('/var'),
            Path('/usr'),
            Path('/bin'),
            Path('/sbin'),
            Path('/root'),
            Path.home() / '.ssh',
            Path.home() / '.gnupg',
            Path.home() / '.aws',
            Path.home() / '.config',
        ]

        for sensitive_dir in sensitive_dirs:
            try:
                if file_path.is_relative_to(sensitive_dir):
                    return f"Error: Access denied. Cannot upload files from {sensitive_dir}"
            except (ValueError, TypeError):
                # is_relative_to raises ValueError if not relative, which is fine
                pass

        if not file_path.exists():
            return f"Error: File not found: {params.file_path}"

        if not file_path.is_file():
            return f"Error: Path is not a file: {params.file_path}"

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > rag_config.max_file_size_mb:
            return f"Error: File too large ({file_size_mb:.1f} MB). Maximum size is {rag_config.max_file_size_mb} MB."

        config = {'display_name': params.display_name} if params.display_name else None
        
        operation = client.file_search_stores.upload_to_file_search_store(
            file=params.file_path,
            file_search_store_name=params.store_name,
            config=config
        )
        
        if params.wait_for_completion:
            operation = _wait_for_operation(client, operation)
        
        file_name = os.path.basename(params.file_path)
        display = params.display_name or file_name
        status = "✅ Completed" if operation.done else "⏳ In Progress"
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "success": True,
                "file_name": file_name,
                "display_name": display,
                "completed": operation.done
            }, indent=2)
        
        return f"""## File Upload {status}

- **File**: {file_name}
- **Display Name**: {display}
- **Store**: `{params.store_name}`
"""
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="upload_text",
    annotations={
        "title": "Upload Text Content",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def upload_text(params: UploadTextInput) -> str:
    """Upload and index text content directly to a File Search store."""
    try:
        client = _get_gemini_client()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(params.content)
            temp_path = f.name
        
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file=temp_path,
                file_search_store_name=params.store_name,
                config={'display_name': params.display_name}
            )
            
            if params.wait_for_completion:
                operation = _wait_for_operation(client, operation)
        finally:
            os.unlink(temp_path)
        
        status = "✅ Completed" if operation.done else "⏳ In Progress"
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "success": True,
                "display_name": params.display_name,
                "content_length": len(params.content),
                "completed": operation.done
            }, indent=2)
        
        return f"""## Text Upload {status}

- **Display Name**: {params.display_name}
- **Content Length**: {len(params.content)} characters
- **Store**: `{params.store_name}`
"""
    except Exception as e:
        return _handle_error(e)


# ============================================================================
# MCP Tools - Search
# ============================================================================

@mcp.tool(
    name="search",
    annotations={
        "title": "Search Documents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def search(params: SearchInput) -> str:
    """Search indexed documents and get AI-generated answers with citations."""
    try:
        client = _get_gemini_client()
        
        file_search_tool = types.Tool(
            file_search=types.FileSearch(file_search_store_names=params.store_names)
        )
        
        response = client.models.generate_content(
            model=params.model,
            contents=params.query,
            config=types.GenerateContentConfig(tools=[file_search_tool])
        )
        
        answer = response.text if hasattr(response, 'text') else str(response)
        
        grounding_metadata = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                grounding_metadata = candidate.grounding_metadata
        
        if params.response_format == ResponseFormat.JSON:
            result = {
                "success": True,
                "query": params.query,
                "answer": answer,
                "model": params.model,
            }
            if params.include_citations:
                result["citations"] = _format_citations_json(grounding_metadata)
            return json.dumps(result, indent=2)
        
        lines = [
            "## Search Results",
            f"**Query**: {params.query}\n",
            "### Answer",
            answer,
        ]
        
        if params.include_citations:
            citations = _format_citations_markdown(grounding_metadata)
            if citations:
                lines.append(citations)
        
        return "\n".join(lines)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="list_files",
    annotations={
        "title": "List Files in Store",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def list_files(params: ListFilesInput) -> str:
    """List all files indexed in a File Search store."""
    try:
        client = _get_gemini_client()
        files = list(client.file_search_stores.list_files(
            name=params.store_name,
            config={'page_size': params.limit}
        ))
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "success": True,
                "store_name": params.store_name,
                "count": len(files),
                "files": [
                    {
                        "name": getattr(f, 'name', 'Unknown'),
                        "display_name": getattr(f, 'display_name', None),
                        "state": str(getattr(f, 'state', 'UNKNOWN')),
                    }
                    for f in files
                ]
            }, indent=2)
        
        if not files:
            return f"## Files in Store\n\n**Store**: `{params.store_name}`\n\nNo files found."
        
        lines = [f"## Files in Store ({len(files)} found)", f"**Store**: `{params.store_name}`\n"]
        for f in files:
            display = getattr(f, 'display_name', 'N/A')
            lines.append(f"- **{display}** (`{getattr(f, 'name', 'Unknown')}`)")
        
        return "\n".join(lines)
    except AttributeError:
        return f"## Files in Store\n\n*File listing not available in current API version.*"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="delete_file",
    annotations={
        "title": "Delete File from Store",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def delete_file(params: DeleteFileInput) -> str:
    """Remove a file from a File Search store."""
    try:
        client = _get_gemini_client()
        client.file_search_stores.delete_file(
            store_name=params.store_name,
            file_name=params.file_name
        )
        return f"## File Deleted\n\n`{params.file_name}` removed from store."
    except AttributeError:
        return "Error: File deletion not available in current API version."
    except Exception as e:
        return _handle_error(e)


# ============================================================================
# MCP Tools - Google Drive Sync
# ============================================================================

@mcp.tool(
    name="sync_drive_folder",
    annotations={
        "title": "Sync Google Drive Folder to Store",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def sync_drive_folder(params: SyncDriveFolderInput) -> str:
    """Sync files from a Google Drive folder to a Gemini File Search store.
    
    This tool fetches files from a Google Drive folder and uploads them to
    a File Search store for indexing. It supports recursive folder traversal
    and handles Google Docs/Sheets/Slides by exporting them.
    
    Args:
        params: SyncDriveFolderInput containing:
            - store_name: Target File Search store
            - folder: Google Drive folder URL or ID (e.g., 'https://drive.google.com/drive/folders/1ABC_xyz' or just '1ABC_xyz')
            - recursive: Include subfolders (default: True)
            - file_extensions: Filter by extensions (optional)
            - max_files: Max files to sync (default: 100)
            - credentials_path: Path to credentials (optional)
    
    Returns:
        str: Sync results including uploaded files and any errors
    """
    if not DRIVE_API_AVAILABLE:
        return """## Error: Google Drive API Not Available

Install the required packages:
```bash
uv sync --extra drive

# Or manually:
uv add google-api-python-client google-auth google-auth-oauthlib
```

**Important:** Google Drive API requires OAuth 2.0 or Service Account authentication.
API keys are NOT supported (unlike Gemini API) because Drive accesses private user data.

Setup options:
1. **Service Account** (recommended for servers/automation):
   - Create in Google Cloud Console → IAM → Service Accounts
   - Download JSON key
   - Share your Drive folder with the service account email
   - Set: `export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'`

2. **OAuth 2.0** (for personal Drive access):
   - Create in Google Cloud Console → APIs → Credentials → OAuth Client ID
   - Download client secrets JSON
   - Set: `export GOOGLE_OAUTH_CREDENTIALS='/path/to/secrets.json'`
   - First run will open browser for consent
"""
    
    try:
        # Parse folder ID from URL or use as-is
        folder_id = _parse_drive_folder_id(params.folder)
        
        # Get Drive service
        service = _get_drive_service(params.credentials_path)
        gemini_client = _get_gemini_client()
        
        # Parse extensions
        extensions = None
        if params.file_extensions:
            extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in params.file_extensions}
        
        # List files in Drive folder
        files = _list_drive_files(
            service,
            folder_id,
            recursive=params.recursive,
            max_files=params.max_files,
            extensions=extensions
        )
        
        if not files:
            return f"""## Sync Complete

No supported files found in folder `{folder_id}`.

Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}
"""
        
        # Create temp directory for downloads
        uploaded = []
        failed = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_info in files:
                file_name = file_info['name']
                
                # Download file
                local_path = _download_drive_file(service, file_info, temp_dir)
                if not local_path:
                    failed.append({"name": file_name, "error": "Download failed"})
                    continue
                
                # Upload to Gemini
                try:
                    operation = gemini_client.file_search_stores.upload_to_file_search_store(
                        file=local_path,
                        file_search_store_name=params.store_name,
                        config={'display_name': file_name}
                    )
                    
                    # Wait for upload
                    operation = _wait_for_operation(gemini_client, operation, max_attempts=30)
                    
                    uploaded.append({
                        "name": file_name,
                        "drive_id": file_info['id'],
                        "status": "indexed" if operation.done else "processing"
                    })
                except Exception as e:
                    failed.append({"name": file_name, "error": str(e)})
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "success": True,
                "store_name": params.store_name,
                "folder_id": folder_id,
                "files_found": len(files),
                "files_uploaded": len(uploaded),
                "files_failed": len(failed),
                "uploaded": uploaded,
                "failed": failed
            }, indent=2)
        
        # Format markdown response
        lines = [
            "## Drive Folder Sync Complete",
            f"**Store**: `{params.store_name}`",
            f"**Folder ID**: `{folder_id}`",
            f"**Files Found**: {len(files)}",
            f"**Successfully Uploaded**: {len(uploaded)}",
            f"**Failed**: {len(failed)}",
        ]
        
        if uploaded:
            lines.append("\n### Uploaded Files")
            for f in uploaded[:20]:  # Show first 20
                lines.append(f"- ✅ {f['name']}")
            if len(uploaded) > 20:
                lines.append(f"- ... and {len(uploaded) - 20} more")
        
        if failed:
            lines.append("\n### Failed Files")
            for f in failed[:10]:
                lines.append(f"- ❌ {f['name']}: {f['error']}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="list_drive_files",
    annotations={
        "title": "List Google Drive Files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def list_drive_files(params: ListDriveFilesInput) -> str:
    """List files in a Google Drive folder that can be indexed.
    
    Preview what files would be synced before running sync_drive_folder.
    
    Args:
        params: ListDriveFilesInput containing:
            - folder: Google Drive folder URL or ID
            - recursive: Include subfolders (default: False)
            - max_files: Max files to list (default: 50)
            - credentials_path: Path to credentials (optional)
    
    Returns:
        str: List of files that can be indexed
    """
    if not DRIVE_API_AVAILABLE:
        return "Error: Google Drive API not available. Install with: uv sync --extra drive"
    
    try:
        # Parse folder ID from URL or use as-is
        folder_id = _parse_drive_folder_id(params.folder)
        
        service = _get_drive_service(params.credentials_path)
        
        files = _list_drive_files(
            service,
            folder_id,
            recursive=params.recursive,
            max_files=params.max_files
        )
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "success": True,
                "folder_id": folder_id,
                "count": len(files),
                "files": [
                    {
                        "id": f['id'],
                        "name": f['name'],
                        "mime_type": f.get('mimeType'),
                        "size": f.get('size'),
                        "modified": f.get('modifiedTime')
                    }
                    for f in files
                ]
            }, indent=2)
        
        if not files:
            return f"## Drive Folder Contents\n\nNo indexable files found in `{folder_id}`."
        
        lines = [
            f"## Drive Folder Contents ({len(files)} indexable files)",
            f"**Folder ID**: `{folder_id}`\n"
        ]
        
        for f in files:
            size = f.get('size', 'N/A')
            if size != 'N/A':
                size = f"{int(size) / 1024:.1f} KB"
            lines.append(f"- **{f['name']}** ({size})")
        
        return "\n".join(lines)
        
    except Exception as e:
        return _handle_error(e)


# ============================================================================
# MCP Tools - RAG Configuration
# ============================================================================

@mcp.tool(
    name="configure_rag",
    annotations={
        "title": "Configure RAG Settings",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def configure_rag(params: ConfigureRAGInput) -> str:
    """Configure RAG pipeline settings.
    
    Set API keys, model parameters, polling behavior, Drive sync options,
    default stores, and response formatting. Only provided values are updated;
    others remain unchanged.
    
    Args:
        params: ConfigureRAGInput with any of:
            - API: gemini_api_key, google_credentials_path
            - Model: model, temperature, max_output_tokens, top_p, top_k
            - Polling: poll_interval_seconds, max_poll_attempts, async_uploads
            - Drive: default_drive_folder, drive_recursive, drive_max_files
            - Store: project_store, default_stores, auto_create_store
            - Retrieval: max_chunks, min_relevance_score
            - Response: include_citations, citation_style, response_format, system_prompt
    
    Returns:
        str: Updated configuration summary
    """
    try:
        updates = {}
        
        # Apply all non-None values
        for field_name, field_value in params.model_dump().items():
            if field_value is not None:
                if hasattr(rag_config, field_name):
                    setattr(rag_config, field_name, field_value)
                    updates[field_name] = field_value
        
        if not updates:
            return "## No Changes\n\nNo configuration values provided. Use `get_rag_config` to view current settings."
        
        # Build response
        lines = ["## RAG Configuration Updated\n", "### Changed Settings"]
        
        for key, value in updates.items():
            # Mask sensitive values
            if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                display_value = "***" + str(value)[-4:] if value else "None"
            elif isinstance(value, str) and len(value) > 50:
                display_value = value[:50] + "..."
            else:
                display_value = value
            
            lines.append(f"- **{key}**: `{display_value}`")
        
        lines.append(f"\n*{len(updates)} setting(s) updated.*")
        
        return "\n".join(lines)
        
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="get_rag_config",
    annotations={
        "title": "Get RAG Configuration",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_rag_config(params: GetRAGConfigInput) -> str:
    """Get current RAG configuration settings.
    
    View all configured settings for the RAG pipeline including model,
    polling, Drive sync, stores, and response options.
    
    Args:
        params: GetRAGConfigInput with:
            - show_sensitive: Show API keys (default: False, masked)
            - response_format: Output format (markdown/json)
    
    Returns:
        str: Current configuration
    """
    try:
        config = rag_config.to_dict()
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"success": True, "config": config}, indent=2)
        
        lines = ["## RAG Configuration\n"]
        
        # Group settings
        groups = {
            "API Configuration": ["gemini_api_key", "google_credentials_path", "google_oauth_path"],
            "Model Settings": ["model", "temperature", "max_output_tokens", "top_p", "top_k"],
            "Polling & Async": ["poll_interval_seconds", "max_poll_attempts", "async_uploads", "batch_size", "concurrent_uploads"],
            "Google Drive": ["default_drive_folder", "drive_recursive", "drive_max_files", "drive_file_extensions", "auto_sync_enabled", "sync_interval_minutes"],
            "Project/Store": ["project_store", "default_stores", "auto_create_store"],
            "Retrieval": ["max_chunks", "min_relevance_score", "include_metadata", "chunk_overlap_context"],
            "Response": ["include_citations", "citation_style", "response_format", "system_prompt"],
            "File Filtering": ["supported_extensions", "max_file_size_mb", "skip_hidden_files"],
        }
        
        for group_name, keys in groups.items():
            lines.append(f"### {group_name}")
            for key in keys:
                if key in config:
                    value = config[key]
                    # Format display
                    if isinstance(value, list) and len(value) > 5:
                        value = f"[{len(value)} items]"
                    elif isinstance(value, str) and len(str(value)) > 60:
                        value = str(value)[:60] + "..."
                    lines.append(f"- **{key}**: `{value}`")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="reset_rag_config",
    annotations={
        "title": "Reset RAG Configuration",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def reset_rag_config(params: ResetRAGConfigInput) -> str:
    """Reset RAG configuration to default values.
    
    Resets all settings to their defaults. Optionally preserves API keys.
    
    Args:
        params: ResetRAGConfigInput with:
            - confirm: Must be True to proceed
            - preserve_api_keys: Keep API keys (default: True)
    
    Returns:
        str: Reset confirmation
    """
    try:
        if not params.confirm:
            return """## Reset Not Confirmed

To reset configuration, set `confirm=true`.

This will reset all RAG settings to defaults.
Set `preserve_api_keys=false` to also clear API credentials.
"""
        
        # Optionally save API keys
        saved_keys = {}
        if params.preserve_api_keys:
            saved_keys = {
                "gemini_api_key": rag_config.gemini_api_key,
                "google_credentials_path": rag_config.google_credentials_path,
                "google_oauth_path": rag_config.google_oauth_path,
            }
        
        # Reset
        rag_config.reset_to_defaults()
        
        # Restore API keys if requested
        if params.preserve_api_keys:
            for key, value in saved_keys.items():
                if value:
                    setattr(rag_config, key, value)
        
        api_status = "preserved" if params.preserve_api_keys else "cleared"
        
        return f"""## RAG Configuration Reset

All settings have been reset to defaults.
API credentials: **{api_status}**

Use `get_rag_config` to view current settings.
Use `configure_rag` to customize settings.
"""
        
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="quick_setup",
    annotations={
        "title": "Quick RAG Setup",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def quick_setup(
    gemini_api_key: Optional[str] = None,
    drive_folder: Optional[str] = None,
    project_name: Optional[str] = None,
    model: str = "gemini-2.5-flash"
) -> str:
    """Quick setup for RAG pipeline with minimal configuration.
    
    One-command setup that configures API key, creates a store,
    and optionally syncs a Google Drive folder.
    
    Args:
        gemini_api_key: Gemini API key (or use GEMINI_API_KEY env var)
        drive_folder: Google Drive folder URL to sync (optional)
        project_name: Name for your project/store (default: 'My RAG Project')
        model: Gemini model to use (default: 'gemini-2.5-flash')
    
    Returns:
        str: Setup summary with next steps
    """
    try:
        results = []
        
        # 1. Configure API key
        if gemini_api_key:
            rag_config.gemini_api_key = gemini_api_key
            results.append("✅ API key configured")
        elif rag_config.gemini_api_key:
            results.append("✅ Using existing API key")
        else:
            return """## Setup Failed

No Gemini API key provided. Either:
1. Pass `gemini_api_key` parameter
2. Set `GEMINI_API_KEY` environment variable

Get your key: https://aistudio.google.com/apikey
"""
        
        # 2. Set model
        rag_config.model = model
        results.append(f"✅ Model set to `{model}`")
        
        # 3. Create store
        project_name = project_name or "My RAG Project"
        client = _get_gemini_client()
        store = client.file_search_stores.create(config={'display_name': project_name})
        rag_config.project_store = store.name
        rag_config.default_stores = [store.name]
        results.append(f"✅ Created store: `{store.name}`")
        
        # 4. Optionally sync Drive folder
        if drive_folder:
            rag_config.default_drive_folder = drive_folder
            results.append(f"✅ Drive folder configured: `{drive_folder}`")
            results.append("   → Run `sync_drive_folder` to index files")
        
        # Build response
        lines = [
            "## Quick Setup Complete\n",
            "### Results",
            *[f"- {r}" for r in results],
            "\n### Configuration",
            f"- **Project Store**: `{rag_config.project_store}`",
            f"- **Model**: `{rag_config.model}`",
            f"- **Drive Folder**: `{rag_config.default_drive_folder or 'Not set'}`",
            "\n### Next Steps",
            "1. Upload files: `upload_file(store_name='...')`",
        ]
        
        if drive_folder:
            lines.append("2. Sync Drive: `sync_drive_folder(store_name='...', folder='...')`")
            lines.append("3. Search: `search(query='your question')`")
        else:
            lines.append("2. Search: `search(query='your question')`")
        
        return "\n".join(lines)
        
    except Exception as e:
        return _handle_error(e)


# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    """Main entry point for the MCP server."""
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not set.")
        print("Set it: export GEMINI_API_KEY='your-key'")
        print("Get key: https://aistudio.google.com/apikey\n")
    
    if not DRIVE_API_AVAILABLE:
        print("Note: Google Drive sync disabled (optional packages not installed).")
        print("Enable with: uv sync --extra drive")
        print("Or: uv add google-api-python-client google-auth google-auth-oauthlib\n")
    
    mcp.run()


if __name__ == "__main__":
    main()
