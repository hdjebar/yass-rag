
"""
Main entry point for the YASS-RAG MCP server.
"""
import argparse
import os
import sys
from pathlib import Path

from . import prompts, resources
from .config import rag_config
from .logging import get_logger
from .server import mcp
from .services.drive import DRIVE_API_AVAILABLE

# Import tools, resources, and prompts to ensure they are registered with the server
# ruff: noqa: F401
from .tools import config, drive, search, store, uploads

logger = get_logger("main")


def config_command(args: argparse.Namespace) -> None:
    """Handle configuration command."""
    env_path = Path(".env")

    if args.key:
        # Update or create .env file
        lines = []
        if env_path.exists():
            with open(env_path) as f:
                lines = f.readlines()

        # Remove existing key if present
        lines = [line for line in lines if not line.strip().startswith("GEMINI_API_KEY=")]

        # Add new key
        lines.append(f"GEMINI_API_KEY='{args.key}'\n")

        with open(env_path, "w") as f:
            f.writelines(lines)

        logger.info(f"Gemini API key configured in {env_path.absolute()}")
        print(f"✅ Gemini API key configured in {env_path.absolute()}")
        return

    if args.show:
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            masked = key[:4] + "*" * (len(key) - 8) + key[-4:] if len(key) > 8 else "***"
            print(f"Current Gemini API Key: {masked}")
        else:
            print("Current Gemini API Key: Not set")
        return

    # If no args, showing help
    print("Use --key <YOUR_KEY> to set the Gemini API key.")


def main() -> None:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="YASS-RAG: Yet Another Simple & Smart RAG")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command (default) - parser created for --help support
    subparsers.add_parser("run", help="Run the MCP server")

    # Config command
    cfg_parser = subparsers.add_parser("config", help="Configure YASS-RAG")
    cfg_parser.add_argument("--key", help="Set the Gemini API Key")
    cfg_parser.add_argument("--show", action="store_true", help="Show current configuration")

    # Check if a subcommand was provided
    if len(sys.argv) > 1 and sys.argv[1] in ["config", "run"]:
        args = parser.parse_args()
        if args.command == "config":
            config_command(args)
            return
        # If 'run', fall through to mcp.run()

    # Default behavior: run the server
    logger.info("Starting YASS-RAG MCP server")

    # Validate environment
    api_key = rag_config.gemini_api_key

    if not api_key:
        logger.warning("GEMINI_API_KEY not set")
        logger.info("Set it using: yass-rag config --key <YOUR_KEY>")
        logger.info("Or: export GEMINI_API_KEY='your-key'")
        logger.info("Get key: https://aistudio.google.com/apikey")

    if not DRIVE_API_AVAILABLE:
        logger.info("Google Drive sync disabled (optional packages not installed)")
        logger.info("Enable with: uv sync --extra drive")

    mcp.run()

if __name__ == "__main__":
    main()
