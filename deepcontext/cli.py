"""Command-line interface for DeepContext."""

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from deepcontext.chunker import ChunkConfig, MarkdownCodeChunker, SemanticChunker
from deepcontext.fetcher import ContentFetcher
from deepcontext.models import SourceType
from deepcontext.store import VectorStore

# Load environment variables from .env file
load_dotenv()

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="deepcontext")
def main() -> None:
    """DeepContext - Semantic chunking service for documents, repos, and webpages."""
    pass


@main.command()
@click.argument("source")
@click.option(
    "--host",
    default="localhost",
    help="Qdrant host",
)
@click.option(
    "--port",
    default=6333,
    type=int,
    help="Qdrant port",
)
@click.option(
    "--collection",
    default="deepcontext_chunks",
    help="Collection name",
)
@click.option(
    "--chunk-size",
    default=1024,
    type=int,
    help="Maximum chunk size in tokens",
)
@click.option(
    "--threshold",
    default=0.7,
    type=float,
    help="Similarity threshold for semantic chunking (0-1)",
)
@click.option(
    "--code-aware/--no-code-aware",
    default=True,
    help="Use code-aware chunking for documentation",
)
def ingest(
    source: str,
    host: str,
    port: int,
    collection: str,
    chunk_size: int,
    threshold: float,
    code_aware: bool,
) -> None:
    """
    Ingest content from a source into the vector store.

    SOURCE can be:

    \b
    - A local file path (e.g., ./docs/README.md)
    - A GitHub URL (e.g., https://github.com/vercel/next.js/blob/canary/docs/...)
    - A Confluence page (e.g., https://company.atlassian.net/wiki/spaces/DOCS/pages/123456)
    - A webpage URL (e.g., https://example.com/docs)

    For Confluence, set CONFLUENCE_EMAIL and CONFLUENCE_TOKEN environment variables.
    """
    # Fetch content
    console.print("[cyan]→[/cyan] Fetching content...")

    with ContentFetcher() as fetcher:
        try:
            document = fetcher.fetch(source)
        except Exception as e:
            console.print(f"[red]Error fetching content: {e}[/red]")
            raise click.Abort()

    console.print(f"[green]✓[/green] Fetched: {document.title}")

    # Chunk content (model download progress will show here)
    console.print("[cyan]→[/cyan] Chunking content semantically...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    config = ChunkConfig(
        chunk_size=chunk_size,
        threshold=threshold,
    )

    if code_aware:
        chunker = MarkdownCodeChunker(config)
    else:
        chunker = SemanticChunker(config)

    try:
        chunks = chunker.chunk_document(document)
    except Exception as e:
        console.print(f"[red]Error chunking content: {e}[/red]")
        raise click.Abort()

    console.print(f"[green]✓[/green] Created {len(chunks)} chunks")

    # Index chunks (model download progress will show here)
    console.print("[cyan]→[/cyan] Indexing in Qdrant...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    store = VectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    try:
        indexed = store.index_chunks(chunks)
        console.print(f"[green]✓[/green] Indexed {indexed} chunks")
    except Exception as e:
        console.print(f"[red]Error indexing chunks: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running:[/yellow]")
        console.print("  podman compose up -d")
        raise click.Abort()
    finally:
        store.close()

    # Show summary
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Successfully ingested[/green]\n\n"
        f"Source: {document.source}\n"
        f"Title: {document.title}\n"
        f"Chunks: {len(chunks)}\n"
        f"Collection: {collection}",
        title="Ingestion Complete",
    ))


@main.command()
@click.argument("repo")
@click.option(
    "--branch",
    default=None,
    help="Branch to fetch from (default: auto-detect)",
)
@click.option(
    "--path",
    default="",
    help="Path within the repo to start from",
)
@click.option(
    "--extensions",
    default=".md,.mdx",
    help="Comma-separated list of file extensions to fetch",
)
@click.option(
    "--host",
    default="localhost",
    help="Qdrant host",
)
@click.option(
    "--port",
    default=6333,
    type=int,
    help="Qdrant port",
)
@click.option(
    "--collection",
    default="deepcontext_chunks",
    help="Collection name",
)
def ingest_repo(
    repo: str,
    branch: str,
    path: str,
    extensions: str,
    host: str,
    port: int,
    collection: str,
) -> None:
    """
    Ingest all documentation from a GitHub repository.

    REPO should be in format "owner/repo" (e.g., vercel/next.js)
    """
    ext_list = [e.strip() for e in extensions.split(",")]

    console.print(f"[cyan]→[/cyan] Fetching {repo}...")

    with ContentFetcher() as fetcher:
        try:
            documents = fetcher.fetch_github_repo(
                repo=repo,
                branch=branch,
                path=path,
                extensions=ext_list,
            )
        except Exception as e:
            console.print(f"[red]Error fetching repository: {e}[/red]")
            raise click.Abort()

    console.print(f"[green]✓[/green] Fetched {len(documents)} documents")

    # Chunk all documents (model download progress will show here)
    console.print("[cyan]→[/cyan] Chunking documents...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    config = ChunkConfig()
    chunker = MarkdownCodeChunker(config)

    all_chunks = []
    for i, doc in enumerate(documents, 1):
        console.print(f"[dim]  [{i}/{len(documents)}] {doc.title}[/dim]")
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            console.print(f"[yellow]Warning: Error chunking {doc.title}: {e}[/yellow]")

    console.print(f"[green]✓[/green] Created {len(all_chunks)} chunks")

    # Index chunks (model download progress will show here)
    console.print("[cyan]→[/cyan] Indexing in Qdrant...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    store = VectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    try:
        indexed = store.index_chunks(all_chunks)
        console.print(f"[green]✓[/green] Indexed {indexed} chunks")
    except Exception as e:
        console.print(f"[red]Error indexing chunks: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running:[/yellow]")
        console.print("  podman compose up -d")
        raise click.Abort()
    finally:
        store.close()

    # Show summary
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Successfully ingested repository[/green]\n\n"
        f"Repository: {repo}\n"
        f"Branch: {branch}\n"
        f"Documents: {len(documents)}\n"
        f"Chunks: {len(all_chunks)}\n"
        f"Collection: {collection}",
        title="Repository Ingestion Complete",
    ))


@main.command()
@click.argument("base_url")
@click.argument("space_key")
@click.option(
    "--limit",
    default=100,
    type=int,
    help="Maximum number of pages to fetch",
)
@click.option(
    "--host",
    default="localhost",
    help="Qdrant host",
)
@click.option(
    "--port",
    default=6333,
    type=int,
    help="Qdrant port",
)
@click.option(
    "--collection",
    default="deepcontext_chunks",
    help="Collection name",
)
def ingest_confluence(
    base_url: str,
    space_key: str,
    limit: int,
    host: str,
    port: int,
    collection: str,
) -> None:
    """
    Ingest all pages from a Confluence space.

    \b
    BASE_URL: Your Confluence base URL (e.g., https://company.atlassian.net/wiki)
    SPACE_KEY: The space key (e.g., DOCS, ENG, TEAM)

    Requires CONFLUENCE_EMAIL and CONFLUENCE_TOKEN environment variables.

    \b
    Example:
        export CONFLUENCE_EMAIL=you@company.com
        export CONFLUENCE_TOKEN=your_api_token
        deepcontext ingest-confluence https://company.atlassian.net/wiki DOCS
    """
    console.print(f"[cyan]→[/cyan] Fetching Confluence space {space_key}...")

    with ContentFetcher() as fetcher:
        try:
            documents = fetcher.fetch_confluence_space(
                base_url=base_url,
                space_key=space_key,
                limit=limit,
            )
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Set CONFLUENCE_EMAIL and CONFLUENCE_TOKEN env vars[/yellow]")
            raise click.Abort()
        except Exception as e:
            console.print(f"[red]Error fetching Confluence space: {e}[/red]")
            raise click.Abort()

    console.print(f"[green]✓[/green] Fetched {len(documents)} pages")

    if not documents:
        console.print("[yellow]No pages found in space.[/yellow]")
        return

    # Chunk all documents
    console.print("[cyan]→[/cyan] Chunking pages...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    config = ChunkConfig()
    chunker = MarkdownCodeChunker(config)

    all_chunks = []
    for i, doc in enumerate(documents, 1):
        console.print(f"[dim]  [{i}/{len(documents)}] {doc.title}[/dim]")
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            console.print(f"[yellow]Warning: Error chunking {doc.title}: {e}[/yellow]")

    console.print(f"[green]✓[/green] Created {len(all_chunks)} chunks")

    # Index chunks
    console.print("[cyan]→[/cyan] Indexing in Qdrant...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    store = VectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    try:
        indexed = store.index_chunks(all_chunks)
        console.print(f"[green]✓[/green] Indexed {indexed} chunks")
    except Exception as e:
        console.print(f"[red]Error indexing chunks: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running:[/yellow]")
        console.print("  podman compose up -d")
        raise click.Abort()
    finally:
        store.close()

    # Show summary
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Successfully ingested Confluence space[/green]\n\n"
        f"Space: {space_key}\n"
        f"Base URL: {base_url}\n"
        f"Pages: {len(documents)}\n"
        f"Chunks: {len(all_chunks)}\n"
        f"Collection: {collection}",
        title="Confluence Ingestion Complete",
    ))


@main.command()
@click.argument("url")
@click.option(
    "--max-pages",
    default=100,
    type=int,
    help="Maximum number of pages to fetch",
)
@click.option(
    "--pattern",
    default=None,
    help="URL pattern to filter pages (regex)",
)
@click.option(
    "--no-sitemap",
    is_flag=True,
    help="Don't try to use sitemap.xml, crawl links instead",
)
@click.option(
    "--host",
    default="localhost",
    help="Qdrant host",
)
@click.option(
    "--port",
    default=6333,
    type=int,
    help="Qdrant port",
)
@click.option(
    "--collection",
    default="deepcontext_chunks",
    help="Collection name",
)
def ingest_website(
    url: str,
    max_pages: int,
    pattern: str | None,
    no_sitemap: bool,
    host: str,
    port: int,
    collection: str,
) -> None:
    """
    Ingest all pages from a website using sitemap or crawling.

    \b
    URL: The base URL to start from (e.g., https://ui.shadcn.com/docs)

    The command will:
    1. Try to find and parse sitemap.xml
    2. If no sitemap, crawl links from the base URL
    3. Filter URLs to only those under the base path

    \b
    Examples:
        # Ingest shadcn docs (uses sitemap automatically)
        deepcontext ingest-website https://ui.shadcn.com/docs

        # Ingest with URL pattern filter
        deepcontext ingest-website https://nextjs.org/docs --pattern "/docs/app/"

        # Force crawling instead of sitemap
        deepcontext ingest-website https://example.com/docs --no-sitemap
    """
    console.print(f"[cyan]→[/cyan] Discovering pages from {url}...")

    def progress_callback(page_url: str, current: int, total: int) -> None:
        console.print(f"[dim]  [{current}/{total}] {page_url[:60]}...[/dim]")

    with ContentFetcher() as fetcher:
        try:
            documents = fetcher.fetch_website(
                base_url=url,
                max_pages=max_pages,
                url_pattern=pattern,
                use_sitemap=not no_sitemap,
                progress_callback=progress_callback,
            )
        except Exception as e:
            console.print(f"[red]Error fetching website: {e}[/red]")
            raise click.Abort()

    console.print(f"[green]✓[/green] Fetched {len(documents)} pages")

    if not documents:
        console.print("[yellow]No pages found.[/yellow]")
        return

    # Chunk all documents
    console.print("[cyan]→[/cyan] Chunking pages...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    config = ChunkConfig()
    chunker = MarkdownCodeChunker(config)

    all_chunks = []
    for i, doc in enumerate(documents, 1):
        console.print(f"[dim]  [{i}/{len(documents)}] {doc.title[:50]}...[/dim]")
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            console.print(f"[yellow]Warning: Error chunking {doc.title}: {e}[/yellow]")

    console.print(f"[green]✓[/green] Created {len(all_chunks)} chunks")

    # Index chunks
    console.print("[cyan]→[/cyan] Indexing in Qdrant...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    store = VectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    try:
        indexed = store.index_chunks(all_chunks)
        console.print(f"[green]✓[/green] Indexed {indexed} chunks")
    except Exception as e:
        console.print(f"[red]Error indexing chunks: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running:[/yellow]")
        console.print("  podman compose up -d")
        raise click.Abort()
    finally:
        store.close()

    # Show summary
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Successfully ingested website[/green]\n\n"
        f"URL: {url}\n"
        f"Pages: {len(documents)}\n"
        f"Chunks: {len(all_chunks)}\n"
        f"Collection: {collection}",
        title="Website Ingestion Complete",
    ))


@main.command()
@click.argument("query")
@click.option(
    "--limit",
    default=5,
    type=int,
    help="Maximum number of results",
)
@click.option(
    "--host",
    default="localhost",
    help="Qdrant host",
)
@click.option(
    "--port",
    default=6333,
    type=int,
    help="Qdrant port",
)
@click.option(
    "--collection",
    default="deepcontext_chunks",
    help="Collection name",
)
@click.option(
    "--language",
    default=None,
    help="Filter by code language",
)
@click.option(
    "--source-type",
    type=click.Choice(["markdown", "github", "webpage", "confluence"]),
    default=None,
    help="Filter by source type",
)
def search(
    query: str,
    limit: int,
    host: str,
    port: int,
    collection: str,
    language: str | None,
    source_type: str | None,
) -> None:
    """
    Search for semantically similar chunks.

    QUERY is your search query (e.g., "middleware in nextjs")
    """
    store = VectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    source_type_enum = SourceType(source_type) if source_type else None

    console.print("[cyan]→[/cyan] Searching...")
    console.print("[dim]  (downloading embedding model if needed...)[/dim]")

    try:
        results = store.search(
            query=query,
            limit=limit,
            source_type=source_type_enum,
            language=language,
        )
        console.print(f"[green]✓[/green] Found {len(results)} results")
    except Exception as e:
        console.print(f"[red]Error searching: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running:[/yellow]")
        console.print("  podman compose up -d")
        raise click.Abort()
    finally:
        store.close()

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Display results
    console.print()
    console.print(f"[bold]Search Results for:[/bold] {query}")
    console.print()

    for i, result in enumerate(results, 1):
        chunk = result.chunk
        score = result.score

        # Create a panel for each result
        console.print(f"[dim]{'─' * 60}[/dim]")
        console.print()
        console.print(f"[bold cyan]### {chunk.title}[/bold cyan]")
        console.print()
        console.print(f"[dim]Source:[/dim] {chunk.source}")
        console.print(f"[dim]Score:[/dim] {score:.4f}")
        console.print()
        console.print(chunk.description)

        # Show code blocks
        for code_block in chunk.code_blocks:
            console.print()
            syntax = Syntax(
                code_block,
                chunk.language or "text",
                theme="monokai",
                line_numbers=False,
            )
            console.print(syntax)

        console.print()

    console.print(f"[dim]{'─' * 60}[/dim]")


@main.command()
@click.option(
    "--host",
    default="localhost",
    help="Qdrant host",
)
@click.option(
    "--port",
    default=6333,
    type=int,
    help="Qdrant port",
)
@click.option(
    "--collection",
    default="deepcontext_chunks",
    help="Collection name",
)
def stats(host: str, port: int, collection: str) -> None:
    """Show statistics about the vector store."""
    store = VectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    try:
        info = store.get_stats()
    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running:[/yellow]")
        console.print("  docker compose up -d")
        raise click.Abort()
    finally:
        store.close()

    table = Table(title="Vector Store Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for key, value in info.items():
        table.add_row(key, str(value))

    console.print(table)


@main.command()
@click.option(
    "--host",
    default="localhost",
    help="Qdrant host",
)
@click.option(
    "--port",
    default=6333,
    type=int,
    help="Qdrant port",
)
@click.option(
    "--collection",
    default="deepcontext_chunks",
    help="Collection name",
)
@click.confirmation_option(prompt="Are you sure you want to clear all data?")
def clear(host: str, port: int, collection: str) -> None:
    """Clear all data from the vector store."""
    store = VectorStore(
        host=host,
        port=port,
        collection_name=collection,
    )

    try:
        store.clear()
        console.print("[green]✓ Collection cleared successfully.[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing collection: {e}[/red]")
        raise click.Abort()
    finally:
        store.close()


if __name__ == "__main__":
    main()

