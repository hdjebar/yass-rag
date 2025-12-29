
"""
File Upload Tools.
"""
import json
import os
import tempfile

from ..config import rag_config
from ..models.api import ResponseFormat, UploadFileInput, UploadTextInput
from ..server import mcp
from ..services.gemini import _get_gemini_client, _wait_for_operation
from ..utils import _handle_error, _validate_file_path


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

        # Validate file path (security + size check)
        validated_path = _validate_file_path(
            params.file_path,
            max_size_mb=rag_config.max_file_size_mb
        )

        config = {'display_name': params.display_name} if params.display_name else None

        operation = client.file_search_stores.upload_to_file_search_store(
            file=str(validated_path),
            file_search_store_name=params.store_name,
            config=config
        )

        if params.wait_for_completion:
            operation = await _wait_for_operation(client, operation)

        file_name = validated_path.name
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
                operation = await _wait_for_operation(client, operation)
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
