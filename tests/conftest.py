"""
Pytest fixtures and configuration.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

# Set test environment variables before importing
os.environ["GEMINI_API_KEY"] = "test-key-12345678"


def pytest_configure(config):
    """Configure pytest."""
    import sys

    # Set MCP mode flag to disable progress bars
    sys._mcp_mode = True


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client."""
    with patch("yass_rag.services.gemini.genai.Client") as mock:
        client = Mock()
        client.file_search_stores = Mock()
        client.operations = Mock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_drive_service():
    """Mock Drive service."""
    with patch("yass_rag.services.drive.build") as mock:
        service = Mock()
        mock.return_value = service
        yield service


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config before each test."""
    from yass_rag.config import rag_config

    original_api_key = rag_config.gemini_api_key
    rag_config.reset_to_defaults()
    rag_config.gemini_api_key = original_api_key
    yield
    rag_config.reset_to_defaults()


@pytest.fixture
def temp_file():
    """Create a temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
        f.write("Test content")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
