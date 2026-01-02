"""
Migration script for moving OAuth tokens from file to keyring.

This script should be run once to migrate existing tokens to secure keyring storage.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from yass_rag.security import store_token, delete_token


def migrate_tokens() -> None:
    """Migrate existing token file to keyring."""
    old_token_file = Path.home() / ".gemini_mcp_token.json"

    if not old_token_file.exists():
        print("ℹ️  No existing token file found. Nothing to migrate.")
        return

    try:
        # Read token from file
        token_json = old_token_file.read_text()

        # Store in keyring
        store_token("drive_oauth", token_json)

        # Delete old file
        old_token_file.unlink()

        print("✅ OAuth token migrated successfully to secure keyring storage.")
        print(f"   Old file removed: {old_token_file}")
        print("\n💡 Your tokens are now stored securely in the OS keyring.")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    print("## YASS-RAG Token Migration\n")

    # Check if keyring is available
    try:
        import keyring
    except ImportError:
        print("❌ keyring package not installed.")
        print("   Run: uv add keyring")
        sys.exit(1)

    # Run migration
    migrate_tokens()

    print("\n✅ Migration complete!")


if __name__ == "__main__":
    main()
