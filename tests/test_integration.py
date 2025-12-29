"""
Integration tests with mocked Gemini API.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from yass_rag.config import rag_config


class MockOperation:
    """Mock for Gemini operation objects."""

    def __init__(self, done: bool = True):
        self.done = done


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


@pytest.fixture
def mock_gemini_client():
    """Create a mocked Gemini client."""
    client = MagicMock()

    # Mock file_search_stores
    client.file_search_stores.create.return_value = MockStore()
    client.file_search_stores.list.return_value = [MockStore()]
    client.file_search_stores.get.return_value = MockStore()
    client.file_search_stores.delete.return_value = None
    client.file_search_stores.upload_to_file_search_store.return_value = MockOperation()
    client.file_search_stores.list_files.return_value = [MockFile()]
    client.file_search_stores.delete_file.return_value = None

    # Mock operations
    client.operations.get.return_value = MockOperation(done=True)

    # Mock models
    mock_response = MagicMock()
    mock_response.text = "This is a test answer based on the documents."
    mock_response.candidates = [
        MagicMock(
            grounding_metadata=MagicMock(
                grounding_chunks=[
                    MagicMock(
                        retrieved_context=MagicMock(
                            title="Test Document", uri="gs://bucket/test.pdf"
                        )
                    )
                ]
            )
        )
    ]
    client.models.generate_content.return_value = mock_response

    return client


@pytest.fixture
def set_api_key():
    """Set API key for tests."""
    original = rag_config.gemini_api_key
    rag_config.gemini_api_key = "test-api-key-12345"
    yield
    rag_config.gemini_api_key = original


class TestStoreTools:
    """Integration tests for store management tools."""

    @pytest.mark.asyncio
    async def test_create_store(self, mock_gemini_client, set_api_key):
        """Test creating a store."""
        with patch(
            "yass_rag.tools.store._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import CreateStoreInput
            from yass_rag.tools.store import create_store

            params = CreateStoreInput(display_name="My Test Store")
            result = await create_store(params)

            assert "File Search Store Created" in result
            assert "My Test Store" in result or "Test Store" in result
            mock_gemini_client.file_search_stores.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_stores(self, mock_gemini_client, set_api_key):
        """Test listing stores."""
        with patch(
            "yass_rag.tools.store._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import ListStoresInput
            from yass_rag.tools.store import list_stores

            params = ListStoresInput(limit=10)
            result = await list_stores(params)

            assert "File Search Stores" in result
            mock_gemini_client.file_search_stores.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_stores_json(self, mock_gemini_client, set_api_key):
        """Test listing stores with JSON response."""
        with patch(
            "yass_rag.tools.store._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import ListStoresInput, ResponseFormat
            from yass_rag.tools.store import list_stores

            params = ListStoresInput(limit=10, response_format=ResponseFormat.JSON)
            result = await list_stores(params)

            data = json.loads(result)
            assert data["success"] is True
            assert "stores" in data

    @pytest.mark.asyncio
    async def test_get_store(self, mock_gemini_client, set_api_key):
        """Test getting store details."""
        with patch(
            "yass_rag.tools.store._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import GetStoreInput
            from yass_rag.tools.store import get_store

            params = GetStoreInput(store_name="fileSearchStores/test123")
            result = await get_store(params)

            assert "File Search Store Details" in result
            mock_gemini_client.file_search_stores.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_store(self, mock_gemini_client, set_api_key):
        """Test deleting a store."""
        with patch(
            "yass_rag.tools.store._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import DeleteStoreInput
            from yass_rag.tools.store import delete_store

            params = DeleteStoreInput(store_name="fileSearchStores/test123")
            result = await delete_store(params)

            assert "Store Deleted" in result
            mock_gemini_client.file_search_stores.delete.assert_called_once()


class TestSearchTools:
    """Integration tests for search tools."""

    @pytest.mark.asyncio
    async def test_search(self, mock_gemini_client, set_api_key):
        """Test searching documents."""
        with patch(
            "yass_rag.tools.search._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import SearchInput
            from yass_rag.tools.search import search

            params = SearchInput(
                store_names=["fileSearchStores/test123"],
                query="What is the main topic?",
            )
            result = await search(params)

            assert "Search Results" in result
            assert "Answer" in result
            mock_gemini_client.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_citations(self, mock_gemini_client, set_api_key):
        """Test searching with citations enabled."""
        with patch(
            "yass_rag.tools.search._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import SearchInput
            from yass_rag.tools.search import search

            params = SearchInput(
                store_names=["fileSearchStores/test123"],
                query="What is the main topic?",
                include_citations=True,
            )
            result = await search(params)

            assert "Search Results" in result
            # Citations should be included if metadata available
            assert "Answer" in result

    @pytest.mark.asyncio
    async def test_list_files(self, mock_gemini_client, set_api_key):
        """Test listing files in a store."""
        with patch(
            "yass_rag.tools.search._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import ListFilesInput
            from yass_rag.tools.search import list_files

            params = ListFilesInput(store_name="fileSearchStores/test123")
            result = await list_files(params)

            assert "Files in Store" in result
            mock_gemini_client.file_search_stores.list_files.assert_called_once()


class TestUploadTools:
    """Integration tests for upload tools."""

    @pytest.mark.asyncio
    async def test_upload_text(self, mock_gemini_client, set_api_key):
        """Test uploading text content."""
        with patch(
            "yass_rag.tools.uploads._get_gemini_client", return_value=mock_gemini_client
        ):
            from yass_rag.models.api import UploadTextInput
            from yass_rag.tools.uploads import upload_text

            params = UploadTextInput(
                store_name="fileSearchStores/test123",
                content="This is test content for indexing.",
                display_name="test-doc.txt",
            )
            result = await upload_text(params)

            assert "Text Upload" in result
            assert "Completed" in result
            mock_gemini_client.file_search_stores.upload_to_file_search_store.assert_called_once()


class TestConfigTools:
    """Integration tests for configuration tools."""

    @pytest.mark.asyncio
    async def test_get_rag_config(self):
        """Test getting RAG configuration."""
        from yass_rag.models.api import GetRAGConfigInput
        from yass_rag.tools.config import get_rag_config

        params = GetRAGConfigInput()
        result = await get_rag_config(params)

        assert "RAG Configuration" in result
        assert "Model Settings" in result
        assert "gemini" in result.lower()

    @pytest.mark.asyncio
    async def test_configure_rag(self):
        """Test configuring RAG settings."""
        from yass_rag.models.api import ConfigureRAGInput
        from yass_rag.tools.config import configure_rag

        original_temp = rag_config.temperature

        params = ConfigureRAGInput(temperature=0.5, max_chunks=15)
        result = await configure_rag(params)

        assert "RAG Configuration Updated" in result
        assert rag_config.temperature == 0.5
        assert rag_config.max_chunks == 15

        # Restore
        rag_config.temperature = original_temp
        rag_config.max_chunks = 10

    @pytest.mark.asyncio
    async def test_reset_rag_config_without_confirm(self):
        """Test that reset requires confirmation."""
        from yass_rag.models.api import ResetRAGConfigInput
        from yass_rag.tools.config import reset_rag_config

        params = ResetRAGConfigInput(confirm=False)
        result = await reset_rag_config(params)

        assert "Reset Not Confirmed" in result

    @pytest.mark.asyncio
    async def test_reset_rag_config_with_confirm(self):
        """Test resetting RAG configuration."""
        from yass_rag.models.api import ResetRAGConfigInput
        from yass_rag.tools.config import reset_rag_config

        # Modify something first
        rag_config.temperature = 0.1

        params = ResetRAGConfigInput(confirm=True, preserve_api_keys=True)
        result = await reset_rag_config(params)

        assert "RAG Configuration Reset" in result
        assert rag_config.temperature == 0.7  # Default value


class TestConcurrency:
    """Tests for concurrency utilities."""

    @pytest.mark.asyncio
    async def test_upload_slot(self):
        """Test upload slot context manager."""
        from yass_rag.concurrency import upload_slot

        async with upload_slot():
            # Should acquire and release slot without error
            pass

    @pytest.mark.asyncio
    async def test_concurrent_uploads(self):
        """Test that concurrent uploads are limited."""
        import asyncio

        from yass_rag.concurrency import _get_upload_semaphore, reset_upload_semaphore, upload_slot

        # Reset to ensure clean state
        reset_upload_semaphore()

        # Set limit to 2
        rag_config.concurrent_uploads = 2
        reset_upload_semaphore()

        semaphore = _get_upload_semaphore()
        assert semaphore._value == 2

        # Test concurrent access
        acquired_count = 0

        async def acquire_slot():
            nonlocal acquired_count
            async with upload_slot():
                acquired_count += 1
                await asyncio.sleep(0.1)
                acquired_count -= 1

        # Start 3 concurrent tasks
        tasks = [acquire_slot() for _ in range(3)]
        await asyncio.gather(*tasks)

        # Restore default
        rag_config.concurrent_uploads = 3
        reset_upload_semaphore()


class TestConfigPersistence:
    """Tests for configuration persistence."""

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        config_path = tmp_path / "test_config.json"

        # Modify config
        rag_config.temperature = 0.3
        rag_config.max_chunks = 20
        rag_config.model = "gemini-2.0-flash"

        # Save
        saved_path = rag_config.save(config_path)
        assert saved_path == config_path
        assert config_path.exists()

        # Reset to defaults
        rag_config.reset_to_defaults()
        assert rag_config.temperature == 0.7
        assert rag_config.max_chunks == 10

        # Load
        loaded = rag_config.load(config_path)
        assert loaded is True
        assert rag_config.temperature == 0.3
        assert rag_config.max_chunks == 20
        assert rag_config.model == "gemini-2.0-flash"

        # Restore defaults
        rag_config.reset_to_defaults()

    def test_load_nonexistent_config(self, tmp_path):
        """Test loading from nonexistent file."""
        config_path = tmp_path / "nonexistent.json"

        loaded = rag_config.load(config_path)
        assert loaded is False

    def test_save_with_secrets(self, tmp_path):
        """Test saving with secrets included."""
        config_path = tmp_path / "test_secrets.json"

        rag_config.gemini_api_key = "test-secret-key"

        # Save without secrets
        rag_config.save(config_path, include_secrets=False)
        with open(config_path) as f:
            data = json.load(f)
        assert "gemini_api_key" not in data

        # Save with secrets
        rag_config.save(config_path, include_secrets=True)
        with open(config_path) as f:
            data = json.load(f)
        assert data["gemini_api_key"] == "test-secret-key"

        # Cleanup
        rag_config.gemini_api_key = None
