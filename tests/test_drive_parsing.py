"""Tests for Google Drive URL parsing functionality."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from yass_rag import _parse_drive_folder_id


class TestParseDriveFolderId:
    """Test cases for _parse_drive_folder_id function."""

    def test_simple_folder_id(self):
        """Test parsing a simple folder ID."""
        folder_id = "1ABC_xyz-123"
        assert _parse_drive_folder_id(folder_id) == folder_id

    def test_basic_url(self):
        """Test parsing a basic Google Drive folder URL."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_with_user_path(self):
        """Test parsing URL with /u/0/ user path."""
        url = "https://drive.google.com/drive/u/0/folders/1ABC_xyz"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_with_different_user_number(self):
        """Test parsing URL with different user number like /u/2/."""
        url = "https://drive.google.com/drive/u/2/folders/1ABC_xyz"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_with_query_params(self):
        """Test parsing URL with query parameters."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz?usp=sharing"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_with_user_and_query_params(self):
        """Test parsing URL with user path and query parameters."""
        url = "https://drive.google.com/drive/u/2/folders/1ABC_xyz?usp=sharing&resourcekey=abc"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_without_https(self):
        """Test parsing URL without https:// prefix."""
        url = "drive.google.com/drive/folders/1ABC_xyz"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_with_http(self):
        """Test parsing URL with http:// prefix."""
        url = "http://drive.google.com/drive/folders/1ABC_xyz"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_with_trailing_slash(self):
        """Test parsing URL with trailing slash."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz/"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_url_with_hash_fragment(self):
        """Test parsing URL with hash fragment."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz#heading=h.abc"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from input."""
        folder_id = "  1ABC_xyz  "
        assert _parse_drive_folder_id(folder_id) == "1ABC_xyz"

    def test_url_with_whitespace(self):
        """Test that whitespace is stripped from URL input."""
        url = "  https://drive.google.com/drive/folders/1ABC_xyz  "
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_long_folder_id(self):
        """Test parsing a longer folder ID (typical format)."""
        folder_id = "1a2B3c4D5e6F7g8H9i0J1k2L3m4N5o6P7"
        assert _parse_drive_folder_id(folder_id) == folder_id

    def test_folder_id_with_underscores_and_hyphens(self):
        """Test folder ID with underscores and hyphens."""
        folder_id = "1ABC_xyz-DEF_123-ghi"
        assert _parse_drive_folder_id(folder_id) == folder_id

    def test_complex_real_world_url(self):
        """Test a complex real-world URL with multiple parameters."""
        url = "https://drive.google.com/drive/u/1/folders/1A2b3C4d5E6f7G8h9I0j?resourcekey=0-abcDEF123&usp=share_link"
        assert _parse_drive_folder_id(url) == "1A2b3C4d5E6f7G8h9I0j"

    def test_mobile_url_format(self):
        """Test mobile-style URL format."""
        url = "https://drive.google.com/drive/mobile/folders/1ABC_xyz"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"

    def test_empty_string(self):
        """Test empty string input."""
        assert _parse_drive_folder_id("") == ""

    def test_only_whitespace(self):
        """Test whitespace-only input."""
        assert _parse_drive_folder_id("   ") == ""

    def test_url_with_special_characters_in_query(self):
        """Test URL with special characters in query string."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz?q=test%20query&key=value"
        assert _parse_drive_folder_id(url) == "1ABC_xyz"


class TestParseDriveFolderIdEdgeCases:
    """Edge case tests for _parse_drive_folder_id function."""

    def test_non_drive_url_with_folders_path(self):
        """Test non-Google URL containing /folders/ path."""
        url = "https://example.com/drive/folders/fake_id"
        # Should still extract from /folders/ pattern
        assert _parse_drive_folder_id(url) == "fake_id"

    def test_folder_id_with_only_numbers(self):
        """Test folder ID containing only numbers."""
        folder_id = "1234567890"
        assert _parse_drive_folder_id(folder_id) == folder_id

    def test_invalid_characters_in_id(self):
        """Test input with invalid characters (not a valid folder ID format)."""
        invalid = "folder with spaces"
        # Should return as-is (let API handle validation)
        assert _parse_drive_folder_id(invalid) == invalid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
