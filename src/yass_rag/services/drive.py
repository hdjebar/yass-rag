"""
Google Drive API service helpers.
"""

import io
import os
import re
import threading
from typing import Any

# Google Drive API (optional - for sync features)
try:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    DRIVE_API_AVAILABLE = True
except ImportError:
    DRIVE_API_AVAILABLE = False

from ..config import DRIVE_SCOPES, GOOGLE_DOCS_EXPORT_MIMES, SUPPORTED_EXTENSIONS
from ..security import retrieve_token, store_token
from ..utils import RateLimiter

# Rate limiter for Drive API calls
drive_rate_limiter = RateLimiter(rate=100, per=60)  # 100 requests per minute

# Service instance cache for connection pooling
_drive_service_pool: dict[str, Any] = {}
_pool_lock = threading.Lock()


def _parse_drive_folder_id(folder_input: str) -> str:
    """Extract folder ID from a Google Drive URL or return the ID if already provided.

    Supports URLs like:
    - https://drive.google.com/drive/folders/1ABC_xyz
    - https://drive.google.com/drive/u/0/folders/1ABC_xyz
    - https://drive.google.com/drive/folders/1ABC_xyz?usp=sharing
    - drive.google.com/drive/folders/1ABC_xyz
    - Just the folder ID: 1ABC_xyz

    Args:
        folder_input: Either a full Google Drive URL or a folder ID

    Returns:
        The extracted folder ID
    """
    folder_input = folder_input.strip()

    # Pattern to match folder ID in various URL formats
    # Matches /folders/{id} where id can contain alphanumeric, underscores, hyphens
    url_pattern = r"(?:drive\.google\.com/.*?/folders/|^)([a-zA-Z0-9_-]+)(?:\?|$|/|#)"

    # If it looks like a URL (contains drive.google.com or starts with http)
    if "drive.google.com" in folder_input or folder_input.startswith("http"):
        match = re.search(url_pattern, folder_input)
        if match:
            return match.group(1)
        # Try simpler extraction - get last path segment before query params
        path = folder_input.split("?")[0].rstrip("/")
        segments = path.split("/")
        if segments:
            return segments[-1]

    # Assume it's already a folder ID
    return folder_input


def _get_drive_credentials(credentials_path: str | None = None):
    """Get Google Drive API credentials (with secure token storage)."""
    if not DRIVE_API_AVAILABLE:
        raise ImportError(
            "Google Drive API not available. Install with: "
            "uv sync --extra drive (or: uv add google-api-python-client google-auth google-auth-oauthlib)"
        )

    creds = None

    # Try credentials path first
    cred_file = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    oauth_file = credentials_path or os.environ.get("GOOGLE_OAUTH_CREDENTIALS")

    # Check for existing token in keyring
    token_json = retrieve_token("drive_oauth")
    if token_json:
        try:
            creds = Credentials.from_authorized_user_info(eval(token_json), DRIVE_SCOPES)
        except Exception:
            pass

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif cred_file and os.path.exists(cred_file):
            # Try service account first
            try:
                creds = service_account.Credentials.from_service_account_file(
                    cred_file, scopes=DRIVE_SCOPES
                )
            except Exception:
                # Fall back to OAuth flow
                flow = InstalledAppFlow.from_client_secrets_file(cred_file, DRIVE_SCOPES)
                creds = flow.run_local_server(port=0)
        elif oauth_file and os.path.exists(oauth_file):
            flow = InstalledAppFlow.from_client_secrets_file(oauth_file, DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        else:
            raise ValueError(
                "No Google credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or "
                "GOOGLE_OAUTH_CREDENTIALS environment variable, or provide credentials_path."
            )

        # Save token to keyring for future use
        if hasattr(creds, "refresh_token") and creds.refresh_token:
            store_token("drive_oauth", str(creds.to_json()))

    return creds


@drive_rate_limiter
def _get_drive_service(credentials_path: str | None = None):
    """Get or create Drive service with pooling (rate-limited)."""
    creds = _get_drive_credentials(credentials_path)

    # Create credential key for pooling
    if hasattr(creds, "service_account_email"):
        creds_key = creds.service_account_email
    elif hasattr(creds, "client_id"):
        creds_key = creds.client_id
    else:
        creds_key = "oauth_user"

    # Check pool first
    with _pool_lock:
        if creds_key not in _drive_service_pool:
            service = build("drive", "v3", credentials=creds)
            _drive_service_pool[creds_key] = service
        return _drive_service_pool[creds_key]


def _clear_drive_service_pool():
    """Clear service pool (for testing/credential changes)."""
    with _pool_lock:
        _drive_service_pool.clear()


@drive_rate_limiter
def _list_drive_files(
    service,
    folder_id: str,
    recursive: bool = True,
    max_files: int = 100,
    extensions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """List files in a Google Drive folder."""
    extensions = extensions or SUPPORTED_EXTENSIONS
    files = []
    folders_to_process = [folder_id]

    while folders_to_process and len(files) < max_files:
        current_folder = folders_to_process.pop(0)

        query = f"'{current_folder}' in parents and trashed = false"
        page_token = None

        while len(files) < max_files:
            response = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageToken=page_token,
                    pageSize=min(100, max_files - len(files)),
                )
                .execute()
            )

            for file in response.get("files", []):
                mime_type = file.get("mimeType", "")

                # Handle folders
                if mime_type == "application/vnd.google-apps.folder":
                    if recursive:
                        folders_to_process.append(file["id"])
                    continue

                # Handle Google Docs (exportable)
                if mime_type in GOOGLE_DOCS_EXPORT_MIMES:
                    files.append(file)
                    continue

                # Check extension for regular files
                name = file.get("name", "")
                ext = os.path.splitext(name)[1].lower()
                if ext in extensions:
                    files.append(file)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return files[:max_files]


def _download_drive_file(
    service, file_info: dict[str, Any], temp_dir: str, chunk_size: int = 1024 * 1024
) -> str | None:
    """Download a file from Google Drive to a temp directory.

    Args:
        service: Drive API service instance
        file_info: File metadata dictionary
        temp_dir: Temporary directory path
        chunk_size: Download chunk size in bytes (default: 1MB)

    Returns:
        Local file path if successful, None otherwise
    """
    file_id = file_info["id"]
    file_name = file_info["name"]
    mime_type = file_info.get("mimeType", "")

    try:
        # Handle Google Docs (need to export)
        if mime_type in GOOGLE_DOCS_EXPORT_MIMES:
            export_mime, ext = GOOGLE_DOCS_EXPORT_MIMES[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
            file_name = os.path.splitext(file_name)[0] + ext
        else:
            request = service.files().get_media(fileId=file_id)

        file_path = os.path.join(temp_dir, file_name)

        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"  Downloading {file_name}: {status.progress() * 100:.1f}%")

        return file_path

    except Exception as e:
        print(f"Warning: Failed to download {file_name}: {e}")
        return None
