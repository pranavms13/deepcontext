"""MCP Server for DeepContext - provides library documentation tools."""

import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from deepcontext.store import LibraryStore, VectorStore, library_id_to_collection_name

# Load environment variables
load_dotenv()


def get_config() -> dict[str, Any]:
    """Get service configuration from environment."""
    return {
        "qdrant_host": os.environ.get("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.environ.get("QDRANT_PORT", "6333")),
    }


# Initialize the MCP server (stateless for HTTP transport)
mcp = FastMCP(
    "DeepContext",
    stateless_http=True,
    instructions="""DeepContext provides semantic documentation search for libraries.
    
Use resolve-library-id to find library IDs before searching documentation.
Use get-library-docs to fetch relevant documentation chunks for a topic.""",
)


@mcp.tool()
def resolve_library_id(libraryName: str) -> str:
    """
    Resolves a package/product name to a deepcontext-compatible library ID and returns a list of matching libraries.

    You MUST call this function before 'get-library-docs' to obtain a valid deepcontext-compatible 
    library ID UNLESS the user explicitly provides a library ID in the format '/org/project' or 
    '/org/project/version' in their query.

    Selection Process:
    1. Analyze the query to understand what library/package the user is looking for
    2. Return the most relevant match based on:
       - Name similarity to the query (exact matches prioritized)
       - Description relevance to the query's intent
       - Documentation coverage (prioritize libraries with higher chunk counts)

    Response Format:
    - Return the selected library ID in a clearly marked section
    - Provide a brief explanation for why this library was chosen
    - If multiple good matches exist, acknowledge this but proceed with the most relevant one
    - If no good matches exist, clearly state this and suggest query refinements

    For ambiguous queries, request clarification before proceeding with a best-guess match.

    Args:
        libraryName: Library name to search for and retrieve a deepcontext-compatible library ID.

    Returns:
        A formatted string containing matching libraries with their IDs and metadata.
    """
    config = get_config()
    
    library_store = LibraryStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
    )
    
    try:
        libraries = library_store.list_libraries(limit=500)
        
        if not libraries:
            return "No libraries found in DeepContext. Please ingest some documentation first."
        
        # Filter libraries by name match (case-insensitive partial match)
        search_term = libraryName.lower()
        matching_libraries = []
        
        for lib in libraries:
            # Check if search term matches library_id, name, or source
            if (search_term in lib.library_id.lower() or 
                search_term in lib.name.lower() or 
                search_term in lib.source.lower()):
                matching_libraries.append(lib)
        
        # If no matches, return all libraries as options
        if not matching_libraries:
            matching_libraries = libraries
            header = f"No exact matches found for '{libraryName}'. Here are all available libraries:\n\n"
        else:
            header = f"Found {len(matching_libraries)} matching libraries for '{libraryName}':\n\n"
        
        # Sort by chunks_count (most documentation first)
        matching_libraries.sort(key=lambda x: x.chunks_count, reverse=True)
        
        # Format output
        result_lines = [header]
        result_lines.append("----------\n")
        
        for lib in matching_libraries[:20]:  # Limit to top 20
            result_lines.append(f"- **Library ID**: `{lib.library_id}`")
            result_lines.append(f"  - Name: {lib.name}")
            result_lines.append(f"  - Source: {lib.source}")
            result_lines.append(f"  - Source Type: {lib.source_type}")
            result_lines.append(f"  - Documents: {lib.documents_count}")
            result_lines.append(f"  - Chunks: {lib.chunks_count}")
            result_lines.append(f"  - Last Updated: {lib.updated_at.isoformat()}")
            result_lines.append("----------\n")
        
        if len(libraries) > 20:
            result_lines.append(f"\n... and {len(libraries) - 20} more libraries available.")
        
        return "\n".join(result_lines)
    
    finally:
        library_store.close()


@mcp.tool()
def get_library_docs(
    deepcontextCompatibleLibraryID: str,
    topic: str = "",
) -> str:
    """
    Fetches up-to-date documentation for a library.

    You must call 'resolve-library-id' first to obtain the exact deepcontext-compatible 
    library ID required to use this tool, UNLESS the user explicitly provides a library ID 
    in the format '/org/project' or '/org/project/version' in their query.

    Args:
        deepcontextCompatibleLibraryID: Exact deepcontext-compatible library ID 
            (e.g., '/mongodb/docs', '/vercel/next.js', '/vercel/next.js/v14.3.0-canary.87') 
            retrieved from 'resolve-library-id' or directly from user query in the format 
            '/org/project' or '/org/project/version'.
        topic: Topic to focus documentation on (e.g., 'hooks', 'routing', 'authentication').
            If empty, returns general documentation overview.

    Returns:
        Formatted documentation chunks matching the query, including code examples and descriptions.
    """
    config = get_config()
    
    # Ensure library_id starts with /
    library_id = deepcontextCompatibleLibraryID
    if not library_id.startswith("/"):
        library_id = "/" + library_id
    
    # Convert library_id to collection name
    collection_name = library_id_to_collection_name(library_id)
    
    # First verify the library exists
    library_store = LibraryStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
    )
    
    try:
        library = library_store.get_library(library_id)
        if not library:
            return f"Library not found: {library_id}\n\nPlease use 'resolve-library-id' to find available libraries."
    finally:
        library_store.close()
    
    # Create vector store for this library's collection
    vector_store = VectorStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        collection_name=collection_name,
    )
    
    try:
        # Build search query
        search_query = topic if topic else library.name
        
        # Search for relevant chunks
        results = vector_store.search(
            query=search_query,
            limit=15,
        )
        
        if not results:
            return f"No documentation found for topic '{topic}' in library '{library_id}'.\n\nTry a different topic or broader search terms."
        
        # Format output similar to Context7
        output_lines = [
            f"# Documentation for {library.name}",
            f"Library ID: `{library_id}`",
            f"Source: {library.source}",
            "",
            f"## Results for: {topic or 'general overview'}",
            "",
        ]
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            output_lines.append(f"### {i}. {chunk.title}")
            output_lines.append("")
            output_lines.append(f"**Source**: {chunk.source}")
            output_lines.append(f"**Relevance Score**: {result.score:.3f}")
            output_lines.append("")
            
            if chunk.description:
                output_lines.append(chunk.description)
                output_lines.append("")
            
            # Add content (truncate if too long)
            content = chunk.content
            if len(content) > 2000:
                content = content[:2000] + "\n... [truncated]"
            
            if content:
                output_lines.append(content)
                output_lines.append("")
            
            # Add code blocks if present
            for code_block in chunk.code_blocks:
                lang = chunk.language or ""
                # Truncate very long code blocks
                if len(code_block) > 1500:
                    code_block = code_block[:1500] + "\n// ... [truncated]"
                output_lines.append(f"```{lang}")
                output_lines.append(code_block)
                output_lines.append("```")
                output_lines.append("")
            
            output_lines.append("---")
            output_lines.append("")
        
        return "\n".join(output_lines)
    
    finally:
        vector_store.close()


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

