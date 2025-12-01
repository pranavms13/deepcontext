"""DeepContext - Semantic chunking service entry point."""

import sys


def main() -> None:
    """Run the DeepContext service."""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Run the API server (includes MCP server at /mcp)
        import uvicorn

        host = "0.0.0.0"
        port = 8000
        reload = False

        # Parse simple args
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--host" and i + 1 < len(args):
                host = args[i + 1]
                i += 2
            elif args[i] == "--port" and i + 1 < len(args):
                port = int(args[i + 1])
                i += 2
            elif args[i] == "--reload":
                reload = True
                i += 1
            else:
                i += 1

        print(f"Starting DeepContext service on {host}:{port}")
        print(f"  - HTTP API: http://{host}:{port}/docs")
        print(f"  - MCP endpoint: http://{host}:{port}/mcp")
        uvicorn.run("deepcontext.api:app", host=host, port=port, reload=reload)
    elif len(sys.argv) > 1 and sys.argv[1] == "mcp":
        # Run standalone MCP server (stdio transport for local tools)
        from deepcontext.mcp_server import main as mcp_main

        mcp_main()
    else:
        # Run the CLI
        from deepcontext.cli import main as cli_main

        cli_main()


if __name__ == "__main__":
    main()
