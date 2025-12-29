"""
Tests for MCP resources and prompts.
"""
import json
from unittest.mock import MagicMock, patch

from yass_rag.config import rag_config


class MockStore:
    """Mock for Gemini store objects."""

    def __init__(self, name: str = "fileSearchStores/test123", display_name: str = "Test Store"):
        self.name = name
        self.display_name = display_name
        self.create_time = "2025-01-01T00:00:00Z"


class MockFile:
    """Mock for Gemini file objects."""

    def __init__(self, name: str = "files/test", display_name: str = "test.pdf"):
        self.name = name
        self.display_name = display_name
        self.state = "ACTIVE"


class TestResources:
    """Tests for MCP resources."""

    def test_get_current_config(self):
        """Test getting current configuration resource."""
        from yass_rag.resources import get_current_config

        result = get_current_config()
        data = json.loads(result)

        # Check key config fields are present
        assert "model" in data
        assert "temperature" in data
        assert "max_chunks" in data
        assert "gemini_api_key" in data  # Should be masked or None

    def test_get_default_config(self):
        """Test getting default configuration resource."""
        from yass_rag.resources import get_default_config

        result = get_default_config()
        data = json.loads(result)

        # Verify default values
        assert data["temperature"] == 0.7
        assert data["max_chunks"] == 10
        assert data["model"] == "gemini-2.5-flash"

    def test_get_model_options(self):
        """Test getting model options resource."""
        from yass_rag.resources import get_model_options

        result = get_model_options()
        data = json.loads(result)

        assert "current_model" in data
        assert "available_models" in data
        assert "gemini-2.5-flash" in data["available_models"]
        assert "temperature_range" in data

    def test_get_supported_extensions(self):
        """Test getting supported extensions resource."""
        from yass_rag.resources import get_supported_extensions

        result = get_supported_extensions()
        data = json.loads(result)

        assert "supported_extensions" in data
        assert "by_category" in data
        assert ".pdf" in data["supported_extensions"]
        assert ".py" in data["supported_extensions"]
        assert "max_file_size_mb" in data

    def test_list_all_stores_success(self):
        """Test listing stores resource with successful API call."""
        mock_client = MagicMock()
        mock_client.file_search_stores.list.return_value = [
            MockStore("fileSearchStores/store1", "Store 1"),
            MockStore("fileSearchStores/store2", "Store 2"),
        ]

        with patch("yass_rag.resources._get_gemini_client", return_value=mock_client):
            from yass_rag.resources import list_all_stores

            result = list_all_stores()
            data = json.loads(result)

            assert data["success"] is True
            assert data["count"] == 2
            assert len(data["stores"]) == 2

    def test_list_all_stores_error(self):
        """Test listing stores resource with API error."""
        mock_client = MagicMock()
        mock_client.file_search_stores.list.side_effect = Exception("API Error")

        with patch("yass_rag.resources._get_gemini_client", return_value=mock_client):
            from yass_rag.resources import list_all_stores

            result = list_all_stores()
            data = json.loads(result)

            assert data["success"] is False
            assert "error" in data

    def test_get_store_details_success(self):
        """Test getting store details resource."""
        mock_client = MagicMock()
        mock_client.file_search_stores.get.return_value = MockStore()
        mock_client.file_search_stores.list_files.return_value = [
            MockFile("files/f1", "doc1.pdf"),
            MockFile("files/f2", "doc2.txt"),
        ]

        with patch("yass_rag.resources._get_gemini_client", return_value=mock_client):
            from yass_rag.resources import get_store_details

            result = get_store_details("fileSearchStores/test123")
            data = json.loads(result)

            assert data["success"] is True
            assert "store" in data
            assert "files" in data
            assert data["files"]["count"] == 2

    def test_get_store_details_error(self):
        """Test getting store details with error."""
        mock_client = MagicMock()
        mock_client.file_search_stores.get.side_effect = Exception("Store not found")

        with patch("yass_rag.resources._get_gemini_client", return_value=mock_client):
            from yass_rag.resources import get_store_details

            result = get_store_details("fileSearchStores/invalid")
            data = json.loads(result)

            assert data["success"] is False
            assert "error" in data


class TestPrompts:
    """Tests for MCP prompts."""

    def test_rag_search_prompt(self):
        """Test RAG search prompt generation."""
        from yass_rag.prompts import rag_search_prompt

        result = rag_search_prompt("What is machine learning?", "store1, store2")

        assert "What is machine learning?" in result
        assert "store1, store2" in result
        assert "search" in result.lower()
        assert "include_citations" in result

    def test_rag_search_prompt_no_stores(self):
        """Test RAG search prompt with no stores specified."""
        from yass_rag.prompts import rag_search_prompt

        result = rag_search_prompt("Test query")

        assert "Test query" in result
        # Should show placeholder when no stores configured
        assert "[specify store names]" in result or len(rag_config.default_stores) > 0

    def test_rag_summarize_prompt(self):
        """Test RAG summarize prompt generation."""
        from yass_rag.prompts import rag_summarize_prompt

        result = rag_summarize_prompt("fileSearchStores/mystore", "technical details")

        assert "fileSearchStores/mystore" in result
        assert "technical details" in result
        assert "summarize" in result.lower()
        assert "list_files" in result

    def test_rag_summarize_prompt_no_focus(self):
        """Test RAG summarize prompt without focus."""
        from yass_rag.prompts import rag_summarize_prompt

        result = rag_summarize_prompt("fileSearchStores/mystore")

        assert "fileSearchStores/mystore" in result
        assert "Focus on" not in result

    def test_rag_qa_prompt_inline_citations(self):
        """Test RAG Q&A prompt with inline citations."""
        from yass_rag.prompts import rag_qa_prompt

        result = rag_qa_prompt("How does X work?", "store1", "inline")

        assert "How does X work?" in result
        assert "store1" in result
        assert "inline" in result.lower()

    def test_rag_qa_prompt_footnote_citations(self):
        """Test RAG Q&A prompt with footnote citations."""
        from yass_rag.prompts import rag_qa_prompt

        result = rag_qa_prompt("Test question", "store1", "footnote")

        assert "numbered footnotes" in result

    def test_rag_qa_prompt_end_citations(self):
        """Test RAG Q&A prompt with end citations."""
        from yass_rag.prompts import rag_qa_prompt

        result = rag_qa_prompt("Test question", "store1", "end")

        assert "List all source documents at the end" in result

    def test_rag_compare_prompt(self):
        """Test RAG compare prompt generation."""
        from yass_rag.prompts import rag_compare_prompt

        result = rag_compare_prompt("AI ethics", "store1, store2, store3")

        assert "AI ethics" in result
        assert "store1, store2, store3" in result
        assert "compare" in result.lower()
        assert "Common findings" in result

    def test_rag_extract_prompt_entities(self):
        """Test RAG extract prompt for entities."""
        from yass_rag.prompts import rag_extract_prompt

        result = rag_extract_prompt("fileSearchStores/mystore", "entities")

        assert "fileSearchStores/mystore" in result
        assert "entities" in result
        assert "named entities" in result.lower()
        assert "JSON" in result

    def test_rag_extract_prompt_dates(self):
        """Test RAG extract prompt for dates."""
        from yass_rag.prompts import rag_extract_prompt

        result = rag_extract_prompt("fileSearchStores/mystore", "dates")

        assert "dates" in result
        assert "deadlines" in result.lower()

    def test_rag_extract_prompt_numbers(self):
        """Test RAG extract prompt for numbers."""
        from yass_rag.prompts import rag_extract_prompt

        result = rag_extract_prompt("fileSearchStores/mystore", "numbers")

        assert "numerical data" in result

    def test_rag_extract_prompt_facts(self):
        """Test RAG extract prompt for facts."""
        from yass_rag.prompts import rag_extract_prompt

        result = rag_extract_prompt("fileSearchStores/mystore", "facts")

        assert "factual claims" in result

    def test_rag_setup_prompt(self):
        """Test RAG setup prompt generation."""
        from yass_rag.prompts import rag_setup_prompt

        result = rag_setup_prompt("My Project", "PDF, Word docs")

        assert "My Project" in result
        assert "PDF, Word docs" in result
        assert "create_store" in result
        assert "configure_rag" in result
        assert "upload_file" in result

    def test_rag_setup_prompt_no_doc_types(self):
        """Test RAG setup prompt without document types."""
        from yass_rag.prompts import rag_setup_prompt

        result = rag_setup_prompt("Simple Project")

        assert "Simple Project" in result
        assert "Document types" not in result


class TestResourcesPromptIntegration:
    """Integration tests for resources and prompts working together."""

    def test_config_resource_matches_prompt_defaults(self):
        """Test that config resource values are used in prompts."""
        from yass_rag.resources import get_current_config, get_model_options

        config_result = json.loads(get_current_config())
        model_result = json.loads(get_model_options())

        # Current model in options should match config
        assert model_result["current_model"] == config_result["model"]
        assert model_result["temperature_range"]["current"] == config_result["temperature"]

    def test_extensions_resource_consistency(self):
        """Test that extensions resource matches config."""
        from yass_rag.resources import get_supported_extensions

        result = json.loads(get_supported_extensions())

        # All config extensions should be in resource
        for ext in rag_config.supported_extensions:
            assert ext in result["supported_extensions"]
