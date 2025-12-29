
"""
Main entry point for the YASS-RAG MCP server.
"""
import argparse
import sys
import os
from pathlib import Path

from .config import rag_config
from .server import mcp
# Import tools to ensure they are registered with the server
from .tools import config, drive, search, store, uploads
from .services.drive import DRIVE_API_AVAILABLE


def config_command(args):
    """Handle configuration command."""
    env_path = Path(".env")
    
    if args.key:
        # Update or create .env file
        lines = []
        if env_path.exists():
            with open(env_path, "r") as f:
                lines = f.readlines()
        
        # Remove existing key if present
        lines = [line for line in lines if not line.strip().startswith("GEMINI_API_KEY=")]
        
        # Add new key
        lines.append(f"GEMINI_API_KEY='{args.key}'\n")
        
        with open(env_path, "w") as f:
            f.writelines(lines)
            
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


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="YASS-RAG: Yet Another Simple & Smart RAG")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run the MCP server")
    
    # Config command
    cfg_parser = subparsers.add_parser("config", help="Configure YASS-RAG")
    cfg_parser.add_argument("--key", help="Set the Gemini API Key")
    cfg_parser.add_argument("--show", action="store_true", help="Show current configuration")
    
    # If no arguments, mimic FastMCP behavior (which normally takes over sys.argv)
    # But since we want to support 'yass-rag run' vs 'yass-rag', we need to be careful.
    # FastMCP uses click/typer internally or looks at sys.argv? 
    # Actually mcp.run() usually blocks and handles stdin/stdout.
    # If we invoke it with 'yass-rag', we want to default to run.
    
    # Check if a subcommand was provided
    if len(sys.argv) > 1 and sys.argv[1] in ["config", "run"]:
        args = parser.parse_args()
        if args.command == "config":
            config_command(args)
            return
        # If 'run', fall through to mcp.run()
    
    # Default behavior: run the server
    # Validate environment
    api_key = rag_config.gemini_api_key
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not set.")
        print("Set it using: yass-rag config --key <YOUR_KEY>")
        print("Or: export GEMINI_API_KEY='your-key'")
        print("Get key: https://aistudio.google.com/apikey\n", file=sys.stderr)
    
    if not DRIVE_API_AVAILABLE:
        print("Note: Google Drive sync disabled (optional packages not installed).", file=sys.stderr)
        print("Enable with: uv sync --extra drive", file=sys.stderr)
    
    # Remove our arguments from sys.argv before passing to mcp.run() if needed,
    # but FastMCP might not use argv if we call run().
    # Actually FastMCP.run() uses uvicorn if transport is sse, or stdio otherwise.
    # It might parse args for transport selection.
    
    mcp.run()

if __name__ == "__main__":
    main()
