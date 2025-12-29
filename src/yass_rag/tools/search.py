
"""
Search and Retrieval Tools.
"""
import json

from google.genai import types

from ..models.api import DeleteFileInput, ListFilesInput, ResponseFormat, SearchInput
from ..server import mcp
from ..services.gemini import _format_citations_json, _format_citations_markdown, _get_gemini_client
from ..utils import _handle_error


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
        return "## Files in Store\n\n*File listing not available in current API version.*"
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
