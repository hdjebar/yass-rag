"""
Test input validation.
"""

import pytest
from pydantic import ValidationError
from yass_rag.models.api import (
    GetStoreInput,
    DeleteStoreInput,
    ConfigureRAGInput,
    SyncDriveFolderInput,
    UploadFileInput,
    SearchInput,
    ListFilesInput,
    ListDriveFilesInput,
)


def test_validate_store_name_valid():
    """Test valid store name formats."""
    valid_names = [
        "fileSearchStores/abc123",
        "fileSearchStores/test-store_123",
    ]
    for name in valid_names:
        GetStoreInput(store_name=name)


def test_validate_store_name_invalid():
    """Test invalid store name formats."""
    invalid_names = [
        "invalid-store",
        "abc123",
        "test/abc",
    ]
    for name in invalid_names:
        with pytest.raises(ValidationError):
            GetStoreInput(store_name=name)


def test_delete_store_input_validation():
    """Test DeleteStoreInput model validation."""
    # Valid
    DeleteStoreInput(store_name="fileSearchStores/abc123")

    # Invalid
    with pytest.raises(ValidationError, match="Invalid store name"):
        DeleteStoreInput(store_name="invalid")


def test_upload_file_input_validation():
    """Test UploadFileInput model validation."""
    # Valid
    UploadFileInput(store_name="fileSearchStores/test", file_path="/tmp/test.txt")

    # Invalid store name
    with pytest.raises(ValidationError, match="Invalid store name"):
        UploadFileInput(store_name="invalid-store", file_path="/tmp/test.txt")


def test_search_input_validation():
    """Test SearchInput model validation."""
    # Valid
    SearchInput(store_names=["fileSearchStores/abc123"], query="test query")

    # Invalid store name
    with pytest.raises(ValidationError, match="Invalid store name"):
        SearchInput(store_names=["invalid-store"], query="test query")


def test_list_files_input_validation():
    """Test ListFilesInput model validation."""
    # Valid
    ListFilesInput(store_name="fileSearchStores/abc123")

    # Invalid store name
    with pytest.raises(ValidationError, match="Invalid store name"):
        ListFilesInput(store_name="invalid")


def test_sync_drive_folder_input_valid_urls():
    """Test valid Drive URL formats."""
    valid_urls = [
        "https://drive.google.com/drive/folders/1ABC_xyz",
        "1ABC_xyz",  # Just folder ID
    ]
    for url in valid_urls:
        SyncDriveFolderInput(store_name="fileSearchStores/test", folder=url)
