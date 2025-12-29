"""
Tests for Pydantic data models.
"""
import pytest
from pydantic import ValidationError

from yass_rag.models.api import (
    CitationStyle,
    ConfigureRAGInput,
    CreateStoreInput,
    ListStoresInput,
    ResponseFormat,
    SearchInput,
    SyncDriveFolderInput,
    UploadFileInput,
)


class TestResponseFormat:
    """Tests for ResponseFormat enum."""

    def test_valid_formats(self):
        """Test that valid formats are accepted."""
        assert ResponseFormat.MARKDOWN == "markdown"
        assert ResponseFormat.JSON == "json"


class TestCreateStoreInput:
    """Tests for CreateStoreInput model."""

    def test_valid_input(self):
        """Test valid store creation input."""
        input_data = CreateStoreInput(display_name="My Store")
        assert input_data.display_name == "My Store"
        assert input_data.response_format == ResponseFormat.MARKDOWN

    def test_empty_name_rejected(self):
        """Test that empty display_name is rejected."""
        with pytest.raises(ValidationError):
            CreateStoreInput(display_name="")

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped from name."""
        input_data = CreateStoreInput(display_name="  My Store  ")
        assert input_data.display_name == "My Store"

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            CreateStoreInput(display_name="Store", unknown_field="value")


class TestListStoresInput:
    """Tests for ListStoresInput model."""

    def test_default_limit(self):
        """Test default pagination limit."""
        input_data = ListStoresInput()
        assert input_data.limit == 20

    def test_limit_bounds(self):
        """Test that limit is bounded between 1 and 100."""
        # Valid limits
        assert ListStoresInput(limit=1).limit == 1
        assert ListStoresInput(limit=100).limit == 100

        # Invalid limits
        with pytest.raises(ValidationError):
            ListStoresInput(limit=0)
        with pytest.raises(ValidationError):
            ListStoresInput(limit=101)


class TestSearchInput:
    """Tests for SearchInput model."""

    def test_valid_search(self):
        """Test valid search input."""
        input_data = SearchInput(
            store_names=["store1", "store2"],
            query="What is the meaning of life?"
        )
        assert len(input_data.store_names) == 2
        assert input_data.query == "What is the meaning of life?"

    def test_empty_query_rejected(self):
        """Test that empty query is rejected."""
        with pytest.raises(ValidationError):
            SearchInput(store_names=["store1"], query="")

    def test_empty_stores_rejected(self):
        """Test that empty store list is rejected."""
        with pytest.raises(ValidationError):
            SearchInput(store_names=[], query="test")

    def test_max_stores_limit(self):
        """Test that max 10 stores are allowed."""
        # Valid: 10 stores
        input_data = SearchInput(
            store_names=[f"store{i}" for i in range(10)],
            query="test"
        )
        assert len(input_data.store_names) == 10

        # Invalid: 11 stores
        with pytest.raises(ValidationError):
            SearchInput(
                store_names=[f"store{i}" for i in range(11)],
                query="test"
            )


class TestUploadFileInput:
    """Tests for UploadFileInput model."""

    def test_valid_upload(self):
        """Test valid file upload input."""
        input_data = UploadFileInput(
            store_name="fileSearchStores/abc123",
            file_path="/path/to/file.pdf"
        )
        assert input_data.wait_for_completion is True

    def test_optional_display_name(self):
        """Test that display_name is optional."""
        input_data = UploadFileInput(
            store_name="store",
            file_path="/path/file.txt"
        )
        assert input_data.display_name is None


class TestSyncDriveFolderInput:
    """Tests for SyncDriveFolderInput model."""

    def test_valid_sync(self):
        """Test valid Drive sync input."""
        input_data = SyncDriveFolderInput(
            store_name="fileSearchStores/abc",
            folder="https://drive.google.com/drive/folders/1ABC_xyz"
        )
        assert input_data.recursive is True
        assert input_data.max_files == 100

    def test_max_files_bounds(self):
        """Test max_files bounds (1-1000)."""
        assert SyncDriveFolderInput(
            store_name="s", folder="f", max_files=1
        ).max_files == 1

        assert SyncDriveFolderInput(
            store_name="s", folder="f", max_files=1000
        ).max_files == 1000

        with pytest.raises(ValidationError):
            SyncDriveFolderInput(store_name="s", folder="f", max_files=0)

        with pytest.raises(ValidationError):
            SyncDriveFolderInput(store_name="s", folder="f", max_files=1001)


class TestConfigureRAGInput:
    """Tests for ConfigureRAGInput model."""

    def test_all_optional(self):
        """Test that all fields are optional."""
        input_data = ConfigureRAGInput()
        assert input_data.model is None
        assert input_data.temperature is None

    def test_temperature_bounds(self):
        """Test temperature validation (0.0-2.0)."""
        assert ConfigureRAGInput(temperature=0.0).temperature == 0.0
        assert ConfigureRAGInput(temperature=2.0).temperature == 2.0

        with pytest.raises(ValidationError):
            ConfigureRAGInput(temperature=-0.1)
        with pytest.raises(ValidationError):
            ConfigureRAGInput(temperature=2.1)

    def test_max_output_tokens_bounds(self):
        """Test max_output_tokens validation."""
        assert ConfigureRAGInput(max_output_tokens=1).max_output_tokens == 1
        assert ConfigureRAGInput(max_output_tokens=8192).max_output_tokens == 8192

        with pytest.raises(ValidationError):
            ConfigureRAGInput(max_output_tokens=0)
        with pytest.raises(ValidationError):
            ConfigureRAGInput(max_output_tokens=8193)

    def test_citation_style_enum(self):
        """Test citation style validation."""
        assert ConfigureRAGInput(citation_style=CitationStyle.INLINE).citation_style == CitationStyle.INLINE
        assert ConfigureRAGInput(citation_style=CitationStyle.FOOTNOTE).citation_style == CitationStyle.FOOTNOTE
        assert ConfigureRAGInput(citation_style=CitationStyle.END).citation_style == CitationStyle.END
