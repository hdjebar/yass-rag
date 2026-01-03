
"""
File Search Store Management Tools.
"""
import json

from ..models.api import (
    CreateStoreInput,
    DeleteStoreInput,
    GetStoreInput,
    ListStoresInput,
    ResponseFormat,
)
from ..server import mcp
from ..services.gemini import _format_store_json, _format_store_markdown, _get_gemini_client
from ..utils import tool_handler


@tool_handler
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


@tool_handler
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


@tool_handler
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
    client = _get_gemini_client()
    store = client.file_search_stores.get(name=params.store_name)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"success": True, "store": _format_store_json(store)}, indent=2)

    return f"## File Search Store Details\n\n{_format_store_markdown(store)}"


@tool_handler
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
    client = _get_gemini_client()
    client.file_search_stores.delete(name=params.store_name)
    return f"## Store Deleted\n\n`{params.store_name}` has been permanently deleted."
