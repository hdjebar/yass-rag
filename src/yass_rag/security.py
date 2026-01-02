"""
Secure credential storage using OS keyring.
"""


import keyring

# Keyring service name for YASS-RAG
KEYRING_SERVICE = "yass-rag"


def store_token(service_name: str, token: str) -> bool:
    """Store OAuth token securely in OS keyring.

    Args:
        service_name: Name of the service (e.g., "drive_oauth")
        token: Token string to store

    Returns:
        True if successful

    Raises:
        RuntimeError: If storage fails
    """
    try:
        keyring.set_password(KEYRING_SERVICE, service_name, token)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to store token for {service_name}: {e}") from e


def retrieve_token(service_name: str) -> str | None:
    """Retrieve OAuth token from OS keyring.

    Args:
        service_name: Name of the service

    Returns:
        Token string if found, None otherwise
    """
    try:
        return keyring.get_password(KEYRING_SERVICE, service_name)
    except Exception:
        return None


def delete_token(service_name: str) -> bool:
    """Delete token from keyring.

    Args:
        service_name: Name of the service

    Returns:
        True if deleted, False if not found
    """
    try:
        keyring.delete_password(KEYRING_SERVICE, service_name)
        return True
    except keyring.errors.PasswordDeleteError:
        return False
