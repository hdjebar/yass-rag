
"""
MCP Resources for YASS-RAG.

Resources provide read-only data that LLMs can access as context.
Unlike tools (which perform actions), resources expose data for LLMs to read.
"""
import json

from .config import rag_config
from .logging import get_logger
from .server import mcp
from .services.gemini import _get_gemini_client

logger = get_logger("resources")


@mcp.resource("config://current")
def get_current_config() -> str:
    """Get the current RAG configuration.

    Returns the active configuration settings including model parameters,
    retrieval settings, and Drive sync options (API keys are masked).
    """
    logger.debug("Resource accessed: config://current")
    config_dict = rag_config.to_dict()
    return json.dumps(config_dict, indent=2)


@mcp.resource("stores://list")
def list_all_stores() -> str:
    """List all available File Search stores.

    Returns a JSON array of store objects with their names, display names,
    and creation timestamps.
    """
    logger.debug("Resource accessed: stores://list")
    try:
        client = _get_gemini_client()
        stores = client.file_search_stores.list()

        store_list = []
        for store in stores:
            store_list.append({
                "name": store.name,
                "display_name": getattr(store, "display_name", None),
                "create_time": getattr(store, "create_time", None),
            })

        return json.dumps({
            "success": True,
            "count": len(store_list),
            "stores": store_list
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"Failed to list stores: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@mcp.resource("stores://{store_name}")
def get_store_details(store_name: str) -> str:
    """Get details of a specific File Search store.

    Args:
        store_name: The store name/ID (e.g., 'fileSearchStores/abc123')

    Returns detailed information about the store including its files.
    """
    logger.debug(f"Resource accessed: stores://{store_name}")
    try:
        client = _get_gemini_client()

        # Get store info
        store = client.file_search_stores.get(name=store_name)

        # Get files in store
        files = list(client.file_search_stores.list_files(name=store_name))

        file_list = []
        for f in files:
            file_list.append({
                "name": f.name,
                "display_name": getattr(f, "display_name", None),
                "state": getattr(f, "state", None),
            })

        return json.dumps({
            "success": True,
            "store": {
                "name": store.name,
                "display_name": getattr(store, "display_name", None),
                "create_time": getattr(store, "create_time", None),
            },
            "files": {
                "count": len(file_list),
                "items": file_list
            }
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"Failed to get store details: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@mcp.resource("config://defaults")
def get_default_config() -> str:
    """Get the default RAG configuration values.

    Returns the default settings that would be applied after a reset.
    """
    logger.debug("Resource accessed: config://defaults")

    # Create a temporary config with defaults
    from .config import RAGConfig
    default_config = RAGConfig()
    config_dict = default_config.to_dict()

    return json.dumps(config_dict, indent=2)


@mcp.resource("config://model-options")
def get_model_options() -> str:
    """Get available model options and their descriptions.

    Returns information about supported Gemini models and their capabilities.
    """
    logger.debug("Resource accessed: config://model-options")

    models = {
        "gemini-2.5-flash": {
            "description": "Fast and efficient model for most RAG tasks",
            "best_for": ["document search", "quick answers", "high throughput"],
            "context_window": "1M tokens"
        },
        "gemini-2.5-pro": {
            "description": "Most capable model for complex reasoning",
            "best_for": ["complex analysis", "detailed synthesis", "nuanced responses"],
            "context_window": "2M tokens"
        },
        "gemini-2.0-flash": {
            "description": "Previous generation fast model",
            "best_for": ["legacy compatibility", "cost optimization"],
            "context_window": "1M tokens"
        }
    }

    return json.dumps({
        "current_model": rag_config.model,
        "available_models": models,
        "temperature_range": {"min": 0.0, "max": 2.0, "current": rag_config.temperature},
        "max_output_tokens": {"min": 1, "max": 8192, "current": rag_config.max_output_tokens}
    }, indent=2)


@mcp.resource("config://extensions")
def get_supported_extensions() -> str:
    """Get list of supported file extensions for indexing.

    Returns all file types that can be uploaded and indexed in stores.
    """
    logger.debug("Resource accessed: config://extensions")

    extensions_by_category = {
        "documents": [".pdf", ".doc", ".docx", ".txt", ".md", ".html", ".htm"],
        "data": [".json", ".csv", ".xlsx", ".xls"],
        "presentations": [".pptx"],
        "code": [
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs",
            ".rb", ".php", ".swift", ".kt", ".scala", ".r", ".sql"
        ]
    }

    return json.dumps({
        "supported_extensions": sorted(rag_config.supported_extensions),
        "by_category": extensions_by_category,
        "max_file_size_mb": rag_config.max_file_size_mb
    }, indent=2)
