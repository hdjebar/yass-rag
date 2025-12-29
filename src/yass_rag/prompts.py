
"""
MCP Prompts for YASS-RAG.

Prompts provide reusable prompt templates that LLMs can use.
They help standardize how the RAG system is queried for common use cases.
"""

from .config import rag_config
from .logging import get_logger
from .server import mcp

logger = get_logger("prompts")


@mcp.prompt("rag-search")
def rag_search_prompt(query: str, store_names: str = "") -> str:
    """Generate a prompt for searching documents in RAG stores.

    Args:
        query: The search query or question to answer
        store_names: Comma-separated list of store names (optional, uses defaults if empty)

    Returns a structured prompt for document search with the search tool.
    """
    logger.debug(f"Prompt generated: rag-search for query='{query[:50]}...'")

    stores = store_names if store_names else ", ".join(rag_config.default_stores) or "[specify store names]"

    return f"""Search the document stores for information to answer this query.

**Query**: {query}

**Stores to search**: {stores}

Please use the `search` tool with:
- store_names: [{stores}]
- query: "{query}"
- include_citations: true

After getting results, synthesize the information into a clear, well-cited answer.
If the documents don't contain relevant information, say so clearly.
"""


@mcp.prompt("rag-summarize")
def rag_summarize_prompt(store_name: str, focus: str = "") -> str:
    """Generate a prompt for summarizing content in a store.

    Args:
        store_name: The store to summarize
        focus: Optional focus area for the summary

    Returns a structured prompt for content summarization.
    """
    logger.debug(f"Prompt generated: rag-summarize for store='{store_name}'")

    focus_instruction = f"\n\n**Focus on**: {focus}" if focus else ""

    return f"""Summarize the key content in the document store.

**Store**: {store_name}{focus_instruction}

Steps:
1. First use `list_files` to see what documents are in the store
2. Then use `search` with a broad query to understand the content
3. Provide a structured summary including:
   - Overview of the document collection
   - Key topics and themes
   - Main insights or takeaways
   - Document types and coverage

Keep the summary concise but comprehensive.
"""


@mcp.prompt("rag-qa")
def rag_qa_prompt(question: str, store_names: str = "", citation_style: str = "inline") -> str:
    """Generate a prompt for Q&A with citations.

    Args:
        question: The question to answer
        store_names: Comma-separated store names (optional)
        citation_style: How to format citations (inline, footnote, or end)

    Returns a structured Q&A prompt with citation requirements.
    """
    logger.debug(f"Prompt generated: rag-qa for question='{question[:50]}...'")

    stores = store_names if store_names else ", ".join(rag_config.default_stores) or "[specify store names]"

    citation_instructions = {
        "inline": "Include citations inline like [Document Name, page X]",
        "footnote": "Use numbered footnotes [1], [2] with references at the end",
        "end": "List all source documents at the end of your answer"
    }

    citation_format = citation_instructions.get(citation_style, citation_instructions["inline"])

    return f"""Answer this question using the document stores as your knowledge base.

**Question**: {question}

**Stores**: {stores}

**Citation Requirements**: {citation_format}

Instructions:
1. Search the stores for relevant information
2. Synthesize a comprehensive answer
3. Include proper citations for all claims
4. If information is missing or unclear, note the limitations
5. If documents conflict, present both perspectives

Aim for accuracy over completeness - only include information you can cite.
"""


@mcp.prompt("rag-compare")
def rag_compare_prompt(topic: str, store_names: str) -> str:
    """Generate a prompt for comparing information across stores.

    Args:
        topic: The topic to compare across stores
        store_names: Comma-separated list of stores to compare

    Returns a prompt for cross-store comparison analysis.
    """
    logger.debug(f"Prompt generated: rag-compare for topic='{topic}'")

    return f"""Compare information about this topic across multiple document stores.

**Topic**: {topic}

**Stores to compare**: {store_names}

Instructions:
1. Search each store separately for the topic
2. Identify key points from each source
3. Create a comparison showing:
   - Common findings across stores
   - Unique insights from each store
   - Any contradictions or differences
   - Relative coverage depth

Present the comparison in a clear table or structured format.
"""


@mcp.prompt("rag-extract")
def rag_extract_prompt(store_name: str, extract_type: str = "entities") -> str:
    """Generate a prompt for extracting structured data from documents.

    Args:
        store_name: The store to extract from
        extract_type: Type of extraction (entities, dates, numbers, facts)

    Returns a prompt for structured data extraction.
    """
    logger.debug(f"Prompt generated: rag-extract type='{extract_type}'")

    extraction_instructions = {
        "entities": "Extract all named entities (people, organizations, locations, products)",
        "dates": "Extract all dates, deadlines, and time references",
        "numbers": "Extract all numerical data (statistics, metrics, measurements, prices)",
        "facts": "Extract key factual claims and assertions"
    }

    instruction = extraction_instructions.get(extract_type, extraction_instructions["entities"])

    return f"""Extract structured information from the document store.

**Store**: {store_name}

**Extraction type**: {extract_type}
**Task**: {instruction}

Instructions:
1. Search the store with broad queries to access content
2. Systematically extract the requested information
3. Present results in a structured format (JSON or table)
4. Include source document references for each item
5. Note any ambiguous or uncertain extractions

Output format:
```json
{{
  "extracted_items": [
    {{"value": "...", "context": "...", "source": "..."}}
  ],
  "total_count": N,
  "notes": "..."
}}
```
"""


@mcp.prompt("rag-setup")
def rag_setup_prompt(project_name: str, document_types: str = "") -> str:
    """Generate a prompt for setting up a new RAG project.

    Args:
        project_name: Name for the new project/store
        document_types: Types of documents to be indexed (optional)

    Returns a walkthrough prompt for RAG setup.
    """
    logger.debug(f"Prompt generated: rag-setup for project='{project_name}'")

    doc_info = f"\n**Document types**: {document_types}" if document_types else ""

    return f"""Help set up a new RAG project with proper configuration.

**Project name**: {project_name}{doc_info}

Setup steps:

1. **Create a store**:
   Use `create_store` with display_name="{project_name}"

2. **Configure settings** (optional):
   Use `configure_rag` to adjust:
   - temperature (0.0-1.0, lower = more precise)
   - max_chunks (5-20, more = broader context)
   - include_citations (true for academic/professional use)

3. **Upload documents**:
   - For local files: use `upload_file`
   - For text content: use `upload_text`
   - For Google Drive: use `sync_drive_folder`

4. **Verify setup**:
   - Use `list_files` to confirm uploads
   - Use `get_store` to check store status

5. **Test search**:
   Run a test query with `search` to verify everything works

Would you like me to proceed with these steps?
"""
