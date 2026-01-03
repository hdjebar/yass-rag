"""
RAG Configuration Tools.
"""

import json

from ..config import rag_config
from ..models.api import ConfigureRAGInput, GetRAGConfigInput, ResetRAGConfigInput, ResponseFormat
from ..security import delete_token
from ..server import mcp
from ..services.drive import (
    DRIVE_API_AVAILABLE,
    _get_drive_service,
    _list_drive_files,
    _parse_drive_folder_id,
)
from ..services.gemini import _get_gemini_client
from ..utils import tool_handler


@tool_handler
@mcp.tool(
    name="configure_rag",
    annotations={
        "title": "Configure RAG Settings",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
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
    with rag_config.transaction():
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
        if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
            display_value = "***" + str(value)[-4:] if value else "None"
        elif isinstance(value, str) and len(value) > 50:
            display_value = value[:50] + "..."
        else:
            display_value = value

        lines.append(f"- **{key}**: `{display_value}`")

    lines.append(f"\n*{len(updates)} setting(s) updated.*")

    return "\n".join(lines)


@tool_handler
@mcp.tool(
    name="get_rag_config",
    annotations={
        "title": "Get RAG Configuration",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
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
    config = rag_config.to_dict()

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"success": True, "config": config}, indent=2)

    lines = ["## RAG Configuration\n"]

    # Group settings
    groups = {
        "API Configuration": ["gemini_api_key", "google_credentials_path", "google_oauth_path"],
        "Model Settings": ["model", "temperature", "max_output_tokens", "top_p", "top_k"],
        "Polling & Async": [
            "poll_interval_seconds",
            "max_poll_attempts",
            "async_uploads",
            "batch_size",
            "concurrent_uploads",
        ],
        "Google Drive": [
            "default_drive_folder",
            "drive_recursive",
            "drive_max_files",
            "drive_file_extensions",
            "auto_sync_enabled",
            "sync_interval_minutes",
        ],
        "Project/Store": ["project_store", "default_stores", "auto_create_store"],
        "Retrieval": [
            "max_chunks",
            "min_relevance_score",
            "include_metadata",
            "chunk_overlap_context",
        ],
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


@tool_handler
@mcp.tool(
    name="reset_rag_config",
    annotations={
        "title": "Reset RAG Configuration",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
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
    if not params.confirm:
        return """## Reset Not Confirmed

To reset configuration, set `confirm=true`.

This will reset all RAG settings to defaults.
Set `preserve_api_keys=false` to also clear API credentials.
"""

    # Optionally save API keys and delete OAuth tokens
    saved_keys = {}
    if params.preserve_api_keys:
        saved_keys = {
            "gemini_api_key": rag_config.gemini_api_key,
            "google_credentials_path": rag_config.google_credentials_path,
            "google_oauth_path": rag_config.google_oauth_path,
        }
    else:
        # Delete OAuth tokens from keyring
        delete_token("drive_oauth")

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


@tool_handler
@mcp.tool(
    name="quick_setup",
    annotations={
        "title": "Quick RAG Setup",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def quick_setup(
    gemini_api_key: str | None = None,
    drive_folder: str | None = None,
    project_name: str | None = None,
    model: str = "gemini-2.5-flash",
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
    store = client.file_search_stores.create(config={"display_name": project_name})
    rag_config.project_store = store.name
    rag_config.default_stores = [store.name]
    results.append(f"✅ Created store: `{store.name}`")

    # 4. Optionally validate and sync Drive folder
    if drive_folder:
        if not DRIVE_API_AVAILABLE:
            results.append(
                "⚠️  Google Drive API not installed. Skipping folder validation. "
                "Install with: uv sync --extra drive"
            )
            rag_config.default_drive_folder = drive_folder
            results.append(f"✅ Drive folder configured: `{drive_folder}`")
        else:
            try:
                folder_id = _parse_drive_folder_id(drive_folder)
                service = _get_drive_service()
                # Quick check if folder is accessible
                _list_drive_files(service, folder_id, recursive=False, max_files=1)
                rag_config.default_drive_folder = drive_folder
                results.append(f"✅ Drive folder verified: `{drive_folder}`")
            except Exception as e:
                return f"""## Setup Failed

Cannot access Google Drive folder: `{drive_folder}`

**Error:** {str(e)}

**Action:**
1. Verify folder ID/URL is correct
2. Check that credentials are properly configured
3. Ensure folder is shared with your service account (if using service account)

Get key: https://aistudio.google.com/apikey
"""

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
