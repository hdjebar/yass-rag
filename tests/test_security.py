"""
Test security utilities.
"""

import pytest
from yass_rag.security import store_token, retrieve_token, delete_token


def test_store_and_retrieve_token():
    """Test storing and retrieving tokens."""
    # Test storing
    result = store_token("test_service", "test_token_123")
    assert result is True

    # Test retrieving
    token = retrieve_token("test_service")
    assert token == "test_token_123"

    # Cleanup
    delete_token("test_service")


def test_delete_nonexistent_token():
    """Test deleting non-existent token."""
    # Should return False, not raise
    result = delete_token("nonexistent_service")
    assert result is False


def test_retrieve_nonexistent_token():
    """Test retrieving non-existent token."""
    token = retrieve_token("nonexistent_service")
    assert token is None


def test_token_roundtrip():
    """Test full roundtrip of token storage."""
    service_name = "roundtrip_test"
    original_token = "test-token-with-special-chars-!@#$%"

    # Store
    store_token(service_name, original_token)

    # Retrieve
    retrieved_token = retrieve_token(service_name)
    assert retrieved_token == original_token

    # Delete
    delete_token(service_name)

    # Verify deleted
    assert retrieve_token(service_name) is None
