# YASS-RAG

**Yet Another Simple & Smart RAG** — An MCP server that exposes Google's Gemini File Search API with **Google Drive sync** capabilities for RAG workflows.

## Features

- **Store Management**: Create, list, get, and delete File Search stores
- **Document Indexing**: Upload local files or text content
- **🆕 Google Drive Sync**: Automatically sync Drive folders to stores
- **Semantic Search**: Query documents with natural language + citations
- **Multiple Formats**: Markdown or JSON responses

## Available Tools

| Tool | Description |
|------|-------------|
| `create_store` | Create a new File Search store |
| `list_stores` | List all stores |
| `get_store` | Get store details |
| `delete_store` | Delete a store |
| `upload_file` | Upload a local file |
| `upload_text` | Upload text content |
| `search` | Search with AI-generated answers |
| `list_files` | List files in a store |
| `delete_file` | Remove a file |
| **`sync_drive_folder`** | **Sync Google Drive folder (URL or ID) to store** |
| **`list_drive_files`** | **Preview Drive folder contents (URL or ID)** |
| **`configure_rag`** | **Configure all RAG settings (API keys, model, polling, etc.)** |
| **`get_rag_config`** | **View current RAG configuration** |
| **`reset_rag_config`** | **Reset configuration to defaults** |
| **`quick_setup`** | **One-command RAG setup** |

## Installation

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/yass-rag
cd yass-rag

# Install core dependencies
uv sync

# Or install with Google Drive support
uv sync --extra drive

# Or install everything (including dev tools)
uv sync --all-extras
```

## Configuration

### Required: Gemini API Key

```bash
export GEMINI_API_KEY='your-gemini-api-key'
```

Get your key: https://aistudio.google.com/apikey

### Optional: Google Drive Credentials

**Important:** Google Drive API does **NOT** support API keys (unlike Gemini). It requires OAuth 2.0 or a Service Account because it accesses private user data.

**Option 1: Service Account (recommended for servers)**
```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'
```
- Best for: automation, servers, shared folders
- Setup: Share your Drive folder with the service account email

**Option 2: OAuth (for personal use)**
```bash
export GOOGLE_OAUTH_CREDENTIALS='/path/to/oauth-credentials.json'
```
- Best for: accessing your personal Drive
- Setup: Requires one-time browser consent flow

#### Setting Up Google Drive API

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project (or select existing)
3. Enable "Google Drive API"
4. Create credentials:
   - **Service Account**: Download JSON key, share folder with service account email
   - **OAuth**: Download client secrets JSON, run once to authenticate

## Usage

### Run the Server

```bash
# Using uv
uv run yass-rag

# Or directly
uv run python yass_rag.py

# Or if installed globally
yass-rag
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yass-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yass-rag", "yass-rag"],
      "env": {
        "GEMINI_API_KEY": "your-key",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/credentials.json"
      }
    }
  }
}
```

Or if installed globally:

```json
{
  "mcpServers": {
    "yass-rag": {
      "command": "yass-rag",
      "env": {
        "GEMINI_API_KEY": "your-key"
      }
    }
  }
}
```

## Example: Sync a Drive Folder

```python
# 1. Create a store
create_store(display_name="Company Docs")
# Returns: fileSearchStores/abc123

# 2. Sync a Drive folder (using full URL)
sync_drive_folder(
    store_name="fileSearchStores/abc123",
    folder="https://drive.google.com/drive/folders/1ABC_your_folder_id_XYZ",
    recursive=True,
    max_files=100
)

# Or just use the folder ID directly
sync_drive_folder(
    store_name="fileSearchStores/abc123",
    folder="1ABC_your_folder_id_XYZ"
)

# 3. Search your documents
search(
    store_names=["fileSearchStores/abc123"],
    query="What is our refund policy?"
)
```

### Supported URL Formats

The `folder` parameter accepts:
- Full URL: `https://drive.google.com/drive/folders/1ABC_xyz`
- URL with user path: `https://drive.google.com/drive/u/0/folders/1ABC_xyz`
- URL with sharing params: `https://drive.google.com/drive/folders/1ABC_xyz?usp=sharing`
- Just the folder ID: `1ABC_xyz`

## Supported File Types

**Documents**: PDF, DOCX, DOC, TXT, MD, HTML
**Data**: JSON, CSV, XLSX, XLS, PPTX
**Code**: PY, JS, TS, Java, C++, Go, Rust, etc.
**Google Docs**: Exported as PDF/CSV automatically

Max file size: 100MB per file

## RAG Configuration

Configure all RAG settings with the `configure_rag` tool:

```python
# Quick one-command setup
quick_setup(
    gemini_api_key="your-key",
    drive_folder="https://drive.google.com/drive/folders/1ABC_xyz",
    project_name="My Knowledge Base",
    model="gemini-2.5-flash"
)

# Or configure individual settings
configure_rag(
    # API Configuration
    gemini_api_key="your-key",
    google_credentials_path="/path/to/credentials.json",
    
    # Model Settings
    model="gemini-2.5-pro",
    temperature=0.7,
    max_output_tokens=4096,
    
    # Polling & Async
    poll_interval_seconds=5,
    max_poll_attempts=60,
    async_uploads=False,
    
    # Google Drive Defaults
    default_drive_folder="https://drive.google.com/drive/folders/...",
    drive_recursive=True,
    drive_max_files=100,
    
    # Project/Store Defaults  
    project_store="fileSearchStores/abc123",
    default_stores=["fileSearchStores/abc123"],
    
    # Response Settings
    include_citations=True,
    citation_style="inline",
    system_prompt="You are a helpful assistant..."
)

# View current config
get_rag_config()

# Reset to defaults (preserves API keys)
reset_rag_config(confirm=True, preserve_api_keys=True)
```

### Configuration Options

| Category | Settings |
|----------|----------|
| **API** | `gemini_api_key`, `google_credentials_path` |
| **Model** | `model`, `temperature`, `max_output_tokens`, `top_p`, `top_k` |
| **Polling** | `poll_interval_seconds`, `max_poll_attempts`, `async_uploads`, `batch_size` |
| **Drive** | `default_drive_folder`, `drive_recursive`, `drive_max_files`, `auto_sync_enabled` |
| **Store** | `project_store`, `default_stores`, `auto_create_store` |
| **Retrieval** | `max_chunks`, `min_relevance_score`, `include_metadata` |
| **Response** | `include_citations`, `citation_style`, `response_format`, `system_prompt` |

## Pricing

| Feature | Cost |
|---------|------|
| Storage | Free |
| Query embeddings | Free |
| Indexing | $0.15 / 1M tokens |
| Generation | Standard Gemini pricing |

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run linter
uv run ruff check .

# Run type checker
uv run mypy yass_rag.py

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Test with MCP Inspector
npx @modelcontextprotocol/inspector uv run yass-rag
```

## Resources

- [Gemini File Search Docs](https://ai.google.dev/gemini-api/docs/file-search)
- [Google Drive API](https://developers.google.com/drive/api)
- [MCP Protocol](https://modelcontextprotocol.io)
- [FastMCP](https://github.com/modelcontextprotocol/python-sdk)
