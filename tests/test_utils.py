"""
Tests for utility functions.
"""
import os
import tempfile

import pytest

from yass_rag.utils import _handle_error, _validate_file_path


class TestValidateFilePath:
    """Tests for _validate_file_path function."""

    def test_valid_file(self):
        """Test validation of a valid file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = _validate_file_path(temp_path)
            assert result.exists()
            assert result.is_file()
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            _validate_file_path("/nonexistent/path/to/file.txt")

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            _validate_file_path("../../../etc/passwd")

    def test_directory_rejected(self):
        """Test that directories are rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="not a file"):
                _validate_file_path(temp_dir)

    def test_file_size_limit(self):
        """Test that files exceeding size limit are rejected."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            # Write 2MB of data
            f.write(b"x" * (2 * 1024 * 1024))
            temp_path = f.name

        try:
            # Should fail with 1MB limit
            with pytest.raises(ValueError, match="exceeds limit"):
                _validate_file_path(temp_path, max_size_mb=1)

            # Should pass with 5MB limit
            result = _validate_file_path(temp_path, max_size_mb=5)
            assert result.exists()
        finally:
            os.unlink(temp_path)


class TestHandleError:
    """Tests for _handle_error function."""

    def test_not_found_error(self):
        """Test 404/not found error formatting."""
        error = Exception("Resource 404 not found")
        result = _handle_error(error)
        assert "not found" in result.lower()

    def test_permission_error(self):
        """Test 403/permission error formatting."""
        error = Exception("403 Forbidden - permission denied")
        result = _handle_error(error)
        assert "Permission denied" in result

    def test_rate_limit_error(self):
        """Test 429/rate limit error formatting."""
        error = Exception("429 rate limit exceeded")
        result = _handle_error(error)
        assert "Rate limit" in result

    def test_api_key_error(self):
        """Test API key error formatting."""
        error = Exception("Invalid API key provided")
        result = _handle_error(error)
        assert "API key" in result

    def test_generic_error(self):
        """Test generic error formatting."""
        error = ValueError("Something went wrong")
        result = _handle_error(error)
        assert "ValueError" in result
        assert "Something went wrong" in result
