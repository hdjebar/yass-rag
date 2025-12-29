
from yass_rag.config import rag_config
from yass_rag.models.api import CreateStoreInput


def test_config_initialization():
    """Test that RAG config is initialized with defaults."""
    assert rag_config.model == "gemini-2.5-flash"
    assert rag_config.max_output_tokens == 2048

def test_model_instantiation():
    """Test that Pydantic models can be instantiated."""
    input_data = CreateStoreInput(display_name="Test Store")
    assert input_data.display_name == "Test Store"
    assert input_data.response_format == "markdown"
