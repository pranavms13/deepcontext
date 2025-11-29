# DeepContext

A semantic chunking service for documents, GitHub repos, webpages, and Confluence. It creates semantically relevant chunks from your content that can be queried later.

## Features

- **Multi-source ingestion**: Markdown files, GitHub repositories, webpages, **Confluence pages**
- **Website crawling**: Ingest entire documentation sites via sitemap or link crawling
- **Semantic chunking**: Intelligent text chunking with code-awareness
- **Code-aware**: Extracts code blocks with context and language detection
- **Vector search**: Powered by Qdrant for fast semantic search
- **Fast embeddings**: Uses FastEmbed (BAAI/bge-small-en-v1.5) for efficient embedding generation

## Quick Start

### 1. Start Qdrant

```bash
# Using Docker
docker compose up -d

# Or using Podman
podman compose up -d
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Ingest content

```bash
# From a local markdown file
uv run deepcontext ingest ./docs/README.md

# From a GitHub URL
uv run deepcontext ingest https://github.com/vercel/next.js/blob/canary/contributing/core/developing.md

# From a webpage
uv run deepcontext ingest https://nextjs.org/docs/app/building-your-application/routing/middleware

# From a Confluence page (requires auth)
export CONFLUENCE_EMAIL=you@company.com
export CONFLUENCE_TOKEN=your_api_token
uv run deepcontext ingest https://company.atlassian.net/wiki/spaces/DOCS/pages/123456/Page-Title

# From an entire Confluence space
uv run deepcontext ingest-confluence https://company.atlassian.net/wiki DOCS --limit 50

# From an entire website (uses sitemap or crawls links)
uv run deepcontext ingest-website https://ui.shadcn.com/docs --max-pages 50
```

### 4. Search

```bash
uv run deepcontext search "middleware authentication"
```

## Example Output

```
Search Results for: how to redirect in middleware

────────────────────────────────────────────────────────────

### Example

Source: /path/to/middleware.md
Score: 0.6693

Example typescript code demonstrating implementation.

┌──────────────────────────────────────────────────────────┐
│ import { NextResponse } from 'next/server'               │
│ import type { NextRequest } from 'next/server'           │
│                                                          │
│ export function middleware(request: NextRequest) {       │
│   return NextResponse.redirect(new URL('/home', ...))    │
│ }                                                        │
│                                                          │
│ export const config = {                                  │
│   matcher: '/about/:path*',                              │
│ }                                                        │
└──────────────────────────────────────────────────────────┘

────────────────────────────────────────────────────────────
```

## CLI Commands

### `deepcontext ingest <source>`

Ingest content from any supported source.

**SOURCE** can be:
- A local file path (e.g., `./docs/README.md`)
- A GitHub URL (e.g., `https://github.com/vercel/next.js/blob/canary/docs/...`)
- A Confluence page (e.g., `https://company.atlassian.net/wiki/spaces/DOCS/pages/123456`)
- A webpage URL (e.g., `https://example.com/docs`)

Options:
- `--chunk-size`: Maximum chunk size in tokens (default: 1024)
- `--threshold`: Similarity threshold for semantic chunking (default: 0.7)
- `--code-aware/--no-code-aware`: Use code-aware chunking (default: enabled)
- `--collection`: Qdrant collection name (default: deepcontext_chunks)
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

### `deepcontext ingest-repo <owner/repo>`

Ingest all documentation from a GitHub repository.

Options:
- `--branch`: Branch to fetch from (default: auto-detect)
- `--path`: Path within the repo to start from
- `--extensions`: File extensions to fetch (default: .md,.mdx)
- `--collection`: Qdrant collection name (default: deepcontext_chunks)
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

**Authentication**: Set `GITHUB_TOKEN` environment variable for higher rate limits (5,000/hour vs 60/hour for unauthenticated requests).

```bash
# Create token at https://github.com/settings/tokens with 'public_repo' scope
export GITHUB_TOKEN=ghp_your_token_here
uv run deepcontext ingest-repo vercel/next.js --path docs
```

### `deepcontext ingest-confluence <base_url> <space_key>`

Ingest all pages from a Confluence space.

Requires environment variables:
- `CONFLUENCE_EMAIL`: Your Atlassian account email
- `CONFLUENCE_TOKEN`: Your Atlassian API token ([create one here](https://id.atlassian.com/manage-profile/security/api-tokens))

Options:
- `--limit`: Maximum pages to fetch (default: 100)
- `--collection`: Qdrant collection name (default: deepcontext_chunks)
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

Example:
```bash
export CONFLUENCE_EMAIL=you@company.com
export CONFLUENCE_TOKEN=your_api_token
uv run deepcontext ingest-confluence https://company.atlassian.net/wiki ENGINEERING
```

### `deepcontext ingest-website <url>`

Ingest all pages from a website using sitemap or link crawling.

The command will:
1. Try to find and parse sitemap.xml
2. If no sitemap, crawl links from the base URL
3. Filter URLs to only those under the base path

Options:
- `--max-pages`: Maximum pages to fetch (default: 100)
- `--pattern`: URL pattern to filter pages (regex)
- `--no-sitemap`: Don't try to use sitemap.xml, crawl links instead
- `--collection`: Qdrant collection name (default: deepcontext_chunks)
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

Examples:
```bash
# Ingest shadcn docs (uses sitemap automatically)
uv run deepcontext ingest-website https://ui.shadcn.com/docs

# Ingest with URL pattern filter
uv run deepcontext ingest-website https://nextjs.org/docs --pattern "/docs/app/"

# Force crawling instead of sitemap
uv run deepcontext ingest-website https://example.com/docs --no-sitemap
```

### `deepcontext search <query>`

Search for semantically similar chunks.

Options:
- `--limit`: Maximum results (default: 5)
- `--language`: Filter by code language
- `--source-type`: Filter by source type (markdown, github, webpage, confluence)
- `--collection`: Qdrant collection name (default: deepcontext_chunks)
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

### `deepcontext stats`

Show statistics about the vector store.

Options:
- `--collection`: Qdrant collection name (default: deepcontext_chunks)
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

### `deepcontext clear`

Clear all data from the vector store. Requires confirmation.

Options:
- `--collection`: Qdrant collection name (default: deepcontext_chunks)
- `--host`: Qdrant host (default: localhost)
- `--port`: Qdrant port (default: 6333)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ContentFetcher │ ──▶ │ DocumentChunker │ ──▶ │   VectorStore   │
│                 │     │                 │     │    (Qdrant)     │
│ - Markdown      │     │ - Code-aware    │     │                 │
│ - GitHub        │     │ - Section-based │     │ - FastEmbed     │
│ - Webpages      │     │ - Smart titles  │     │ - Cosine sim    │
│ - Websites      │     │                 │     │                 │
│ - Confluence    │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Python API

```python
from deepcontext import ContentFetcher, VectorStore
from deepcontext.chunker import ChunkConfig, DocumentChunker

# Fetch content
with ContentFetcher() as fetcher:
    doc = fetcher.fetch("https://github.com/vercel/next.js/blob/canary/docs/...")

# Chunk with code awareness
config = ChunkConfig(
    max_chunk_size=2000,  # Max characters per chunk
    min_chunk_size=200,   # Min characters for a chunk
    overlap_size=100,     # Overlap between chunks
)
chunker = DocumentChunker(config)
chunks = chunker.chunk_document(doc)

# Store and search
store = VectorStore()
store.index_chunks(chunks)

results = store.search("middleware authentication")
for result in results:
    print(result.chunk.to_display_format())
    print(f"Score: {result.score}")

store.close()
```

### Fetching multiple documents

```python
from deepcontext import ContentFetcher

with ContentFetcher() as fetcher:
    # Fetch entire GitHub repo
    docs = fetcher.fetch_github_repo("vercel/next.js", path="docs")
    
    # Fetch Confluence space
    docs = fetcher.fetch_confluence_space(
        base_url="https://company.atlassian.net/wiki",
        space_key="DOCS",
        limit=50,
    )
    
    # Fetch entire website
    docs = fetcher.fetch_website(
        base_url="https://ui.shadcn.com/docs",
        max_pages=100,
        url_pattern=r"/docs/",  # Optional regex filter
    )
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub personal access token for higher API rate limits |
| `CONFLUENCE_EMAIL` | Atlassian account email for Confluence access |
| `CONFLUENCE_TOKEN` | Atlassian API token for Confluence access |

You can also use a `.env` file in the project root - it will be loaded automatically.

## Requirements

- Python 3.12+
- Docker or Podman (for Qdrant)

## License

MIT
