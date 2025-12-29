"""
Tests for configuration management.
"""
import pytest

from yass_rag.config import DEFAULT_MODEL, SUPPORTED_EXTENSIONS, RAGConfig


class TestRAGConfig:
    """Tests for RAGConfig class."""

    def test_default_values(self):
        """Test that config is initialized with correct defaults."""
        config = RAGConfig()

        # Model defaults
        assert config.model == DEFAULT_MODEL
        assert config.temperature == 0.7
        assert config.max_output_tokens == 2048
        assert config.top_p == 0.95
        assert config.top_k == 40

        # Polling defaults
        assert config.poll_interval_seconds == 5
        assert config.max_poll_attempts == 60
        assert config.async_uploads is False

        # Response defaults
        assert config.include_citations is True
        assert config.citation_style == "inline"
        assert config.response_format == "markdown"

    def test_reset_to_defaults(self):
        """Test that reset_to_defaults restores default values."""
        config = RAGConfig()

        # Modify values
        config.model = "custom-model"
        config.temperature = 0.1
        config.include_citations = False

        # Reset
        config.reset_to_defaults()

        # Verify defaults restored
        assert config.model == DEFAULT_MODEL
        assert config.temperature == 0.7
        assert config.include_citations is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RAGConfig()
        config.gemini_api_key = "test-key-12345"

        result = config.to_dict()

        # API key should be masked
        assert result["gemini_api_key"] == "***"

        # Other values should be present
        assert result["model"] == DEFAULT_MODEL
        assert "temperature" in result
        assert "max_output_tokens" in result

    def test_from_dict(self):
        """Test updating config from dictionary."""
        config = RAGConfig()

        config.from_dict({
            "model": "new-model",
            "temperature": 0.5,
            "include_citations": False,
        })

        assert config.model == "new-model"
        assert config.temperature == 0.5
        assert config.include_citations is False

    def test_from_dict_with_extensions_list(self):
        """Test that extensions list is converted to set."""
        config = RAGConfig()

        config.from_dict({
            "supported_extensions": [".pdf", ".txt", ".md"]
        })

        assert isinstance(config.supported_extensions, set)
        assert ".pdf" in config.supported_extensions
        assert ".txt" in config.supported_extensions

    def test_get_effective_api_key_raises_when_missing(self):
        """Test that get_effective_api_key raises when no key is set."""
        config = RAGConfig()
        config.gemini_api_key = None

        with pytest.raises(ValueError, match="GEMINI_API_KEY not configured"):
            config.get_effective_api_key()

    def test_get_effective_api_key_returns_key(self):
        """Test that get_effective_api_key returns the configured key."""
        config = RAGConfig()
        config.gemini_api_key = "test-api-key"

        result = config.get_effective_api_key()
        assert result == "test-api-key"


class TestSupportedExtensions:
    """Tests for supported file extensions."""

    def test_common_extensions_supported(self):
        """Test that common file types are supported."""
        common = [".pdf", ".docx", ".txt", ".md", ".py", ".js", ".json"]
        for ext in common:
            assert ext in SUPPORTED_EXTENSIONS, f"{ext} should be supported"

    def test_extensions_are_lowercase(self):
        """Test that all extensions are lowercase."""
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower(), f"{ext} should be lowercase"

    def test_extensions_start_with_dot(self):
        """Test that all extensions start with a dot."""
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith("."), f"{ext} should start with '.'"
