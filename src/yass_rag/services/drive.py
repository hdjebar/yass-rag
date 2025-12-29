
"""
Google Drive API service helpers.
"""
import io
import os
import re
from pathlib import Path
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
    url_pattern = r'(?:drive\.google\.com/.*?/folders/|^)([a-zA-Z0-9_-]+)(?:\?|$|/|#)'

    # If it looks like a URL (contains drive.google.com or starts with http)
    if 'drive.google.com' in folder_input or folder_input.startswith('http'):
        match = re.search(url_pattern, folder_input)
        if match:
            return match.group(1)
        # Try simpler extraction - get last path segment before query params
        path = folder_input.split('?')[0].rstrip('/')
        segments = path.split('/')
        if segments:
            return segments[-1]

    # Assume it's already a folder ID
    return folder_input


def _get_drive_credentials(credentials_path: str | None = None):
    """Get Google Drive API credentials."""
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
        creds = Credentials.from_authorized_user_file(str(token_file), DRIVE_SCOPES)

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

        # Save token for future use with secure permissions
        if hasattr(creds, 'refresh_token') and creds.refresh_token:
            with open(token_file, 'w') as f:
                f.write(creds.to_json())
            # Restrict file permissions to owner only (security)
            os.chmod(token_file, 0o600)

    return creds


def _get_drive_service(credentials_path: str | None = None):
    """Get Google Drive API service."""
    creds = _get_drive_credentials(credentials_path)
    return build('drive', 'v3', credentials=creds)


def _list_drive_files(
    service,
    folder_id: str,
    recursive: bool = True,
    max_files: int = 100,
    extensions: set[str] | None = None
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


def _download_drive_file(service, file_info: dict[str, Any], temp_dir: str) -> str | None:
    """Download a file from Google Drive to a temp directory."""
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
