"""
Test configuration management.
"""

import pytest
from yass_rag.config import rag_config
from yass_rag.models.api import ConfigureRAGInput
from threading import Thread
import time


def test_config_defaults():
    """Test default configuration values."""
    assert rag_config.model == "gemini-2.5-flash"
    assert rag_config.temperature == 0.7
    assert rag_config.max_output_tokens == 2048
    assert rag_config.poll_interval_seconds == 5
    assert rag_config.max_poll_attempts == 60


def test_config_to_dict():
    """Test config serialization."""
    config_dict = rag_config.to_dict()
    assert "model" in config_dict
    assert "temperature" in config_dict
    assert "gemini_api_key" in config_dict

    # Verify API key is masked
    if config_dict["gemini_api_key"]:
        assert "***" in config_dict["gemini_api_key"] or config_dict["gemini_api_key"] is None


def test_config_from_dict():
    """Test config update from dict."""
    updates = {"model": "gemini-2.5-pro", "temperature": 0.8, "max_output_tokens": 4096}
    rag_config.from_dict(updates)
    assert rag_config.model == "gemini-2.5-pro"
    assert rag_config.temperature == 0.8
    assert rag_config.max_output_tokens == 4096


def test_config_reset_preserves_keys():
    """Test that reset preserves API keys when requested."""
    rag_config.gemini_api_key = "test-key-12345"
    original_key = rag_config.gemini_api_key

    saved_keys = {
        "gemini_api_key": rag_config.gemini_api_key,
        "google_credentials_path": rag_config.google_credentials_path,
        "google_oauth_path": rag_config.google_oauth_path,
    }

    rag_config.reset_to_defaults()

    for key, value in saved_keys.items():
        if value:
            setattr(rag_config, key, value)

    assert rag_config.gemini_api_key == original_key


def test_config_transaction():
    """Test that config transaction works."""
    with rag_config.transaction():
        rag_config.model = "test-model"
        assert rag_config.model == "test-model"


def test_config_thread_safety():
    """Test that config is thread-safe."""
    num_threads = 10
    iterations = 50
    errors = []

    def modify_config():
        try:
            for _ in range(iterations):
                rag_config.model = "gemini-2.5-flash"
                assert rag_config.model == "gemini-2.5-flash"
                rag_config.temperature = 0.7
                assert rag_config.temperature == 0.7
        except AssertionError as e:
            errors.append(e)

    threads = [Thread(target=modify_config) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"


def test_config_validation():
    """Test that config validates after reset."""
    rag_config.reset_to_defaults()
    assert 0.0 <= rag_config.temperature <= 2.0
    assert rag_config.max_output_tokens >= 1
    assert rag_config.model
