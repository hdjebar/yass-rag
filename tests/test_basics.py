import pytest

from yass_rag.config import rag_config
from yass_rag.models.api import CreateStoreInput, UploadFileInput
from yass_rag.utils import _handle_error


def test_config_initialization():
    """Test that RAG config is initialized with defaults."""
    assert rag_config.model == "gemini-2.5-flash"
    assert rag_config.max_output_tokens == 2048


def test_model_instantiation():
    """Test that Pydantic models can be instantiated."""
    input_data = CreateStoreInput(display_name="Test Store")
    assert input_data.display_name == "Test Store"
    assert input_data.response_format == "markdown"


def test_error_handling_404():
    """Test 404 error formatting."""
    e = Exception("404 not found")
    result = _handle_error(e)
    assert "Resource Not Found" in result
    assert "Check:" in result


def test_error_handling_permission():
    """Test permission error formatting."""
    e = Exception("403 permission denied")
    result = _handle_error(e)
    assert "Access Denied" in result


def test_error_handling_rate_limit():
    """Test rate limit error formatting."""
    e = Exception("429 rate limit exceeded")
    result = _handle_error(e)
    assert "Rate Limit Exceeded" in result


def test_error_handling_timeout():
    """Test timeout error formatting."""
    from yass_rag.utils import _handle_error

    e = TimeoutError("operation timed out")
    result = _handle_error(e)
    assert "Operation Timed Out" in result


def test_error_handling_generic():
    """Test generic error formatting."""
    e = ValueError("some generic error")
    result = _handle_error(e)
    assert "Error (ValueError)" in result
    assert "some generic error" in result


def test_upload_file_validation():
    """Test UploadFileInput validation."""
    UploadFileInput(store_name="fileSearchStores/test", file_path="/tmp/test.txt")


def test_upload_file_invalid_store():
    """Test UploadFileInput rejects invalid store name."""
    with pytest.raises(Exception):  # ValidationError
        UploadFileInput(store_name="invalid-store", file_path="/tmp/test.txt")
