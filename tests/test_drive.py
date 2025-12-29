"""
Tests for Google Drive service functions.
"""
import pytest


class TestParseDriveFolderId:
    """Tests for _parse_drive_folder_id function."""

    @pytest.fixture
    def parse_folder_id(self):
        """Import the function, skipping if Drive API not available."""
        try:
            from yass_rag.services.drive import _parse_drive_folder_id
            return _parse_drive_folder_id
        except ImportError:
            pytest.skip("Google Drive API not available")

    def test_full_url(self, parse_folder_id):
        """Test parsing full Google Drive URL."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz123"
        assert parse_folder_id(url) == "1ABC_xyz123"

    def test_url_with_user_path(self, parse_folder_id):
        """Test URL with user path segment."""
        url = "https://drive.google.com/drive/u/0/folders/1ABC_xyz123"
        assert parse_folder_id(url) == "1ABC_xyz123"

    def test_url_with_query_params(self, parse_folder_id):
        """Test URL with query parameters."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz123?usp=sharing"
        assert parse_folder_id(url) == "1ABC_xyz123"

    def test_url_without_protocol(self, parse_folder_id):
        """Test URL without https:// prefix."""
        url = "drive.google.com/drive/folders/1ABC_xyz123"
        assert parse_folder_id(url) == "1ABC_xyz123"

    def test_just_folder_id(self, parse_folder_id):
        """Test when only folder ID is provided."""
        folder_id = "1ABC_xyz123-test"
        assert parse_folder_id(folder_id) == "1ABC_xyz123-test"

    def test_whitespace_stripped(self, parse_folder_id):
        """Test that whitespace is stripped."""
        url = "  1ABC_xyz123  "
        assert parse_folder_id(url) == "1ABC_xyz123"

    def test_url_with_hash(self, parse_folder_id):
        """Test URL with hash fragment."""
        url = "https://drive.google.com/drive/folders/1ABC_xyz123#section"
        assert parse_folder_id(url) == "1ABC_xyz123"

    def test_complex_folder_id(self, parse_folder_id):
        """Test folder ID with underscores and hyphens."""
        folder_id = "1a-B_c2-D_e3"
        assert parse_folder_id(folder_id) == "1a-B_c2-D_e3"
