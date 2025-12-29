
"""
Google Drive Sync Tools.
"""
import json
import tempfile

from ..models.api import ListDriveFilesInput, ResponseFormat, SyncDriveFolderInput
from ..server import mcp
from ..services.drive import (
    DRIVE_API_AVAILABLE,
    SUPPORTED_EXTENSIONS,
    _download_drive_file,
    _get_drive_service,
    _list_drive_files,
    _parse_drive_folder_id,
)
from ..services.gemini import _get_gemini_client, _wait_for_operation
from ..utils import _handle_error


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
                    operation = await _wait_for_operation(gemini_client, operation, max_attempts=30)

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
