"""FastAPI web service for DeepContext."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.routing import Mount

from deepcontext.models import IngestJob, JobStatus, JobType, SearchResult, SourceType
from deepcontext.queue import generate_job_id, JobQueue, JobWorker
from deepcontext.store import derive_library_id, library_id_to_collection_name, LibraryStore, VectorStore
from deepcontext.mcp_server import mcp as mcp_server

# Load environment variables
load_dotenv()


# =============================================================================
# Request/Response Models
# =============================================================================


class IngestRequest(BaseModel):
    """Request to ingest a single source."""

    source: str = Field(..., description="URL, file path, or other source to ingest")
    chunk_size: int = Field(default=1024, description="Maximum chunk size in tokens")
    threshold: float = Field(default=0.7, ge=0, le=1, description="Similarity threshold for chunking")
    code_aware: bool = Field(default=True, description="Use code-aware chunking")


class IngestRepoRequest(BaseModel):
    """Request to ingest a GitHub repository."""

    repo: str = Field(..., description="Repository in format 'owner/repo' or full GitHub URL")
    branch: str | None = Field(default=None, description="Branch name (auto-detect if not specified)")
    path: str = Field(default="", description="Path within repo to start from")
    extensions: str = Field(default=".md,.mdx", description="Comma-separated file extensions")


class IngestWebsiteRequest(BaseModel):
    """Request to ingest a website."""

    url: str = Field(..., description="Base URL to start crawling from")
    max_pages: int = Field(default=100, ge=1, le=1000, description="Maximum pages to fetch")
    url_pattern: str | None = Field(default=None, description="Regex pattern to filter URLs")
    use_sitemap: bool = Field(default=True, description="Try to use sitemap.xml")


class IngestConfluenceRequest(BaseModel):
    """Request to ingest a Confluence space."""

    base_url: str = Field(..., description="Confluence base URL")
    space_key: str = Field(..., description="Space key (e.g., DOCS, ENG)")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum pages to fetch")


class SearchRequest(BaseModel):
    """Request to search for chunks."""

    query: str = Field(..., min_length=1, description="Search query")
    library_id: str = Field(..., description="Library ID to search (e.g., /better-auth/better-auth)")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    source_type: SourceType | None = Field(default=None, description="Filter by source type")
    language: str | None = Field(default=None, description="Filter by code language")


class JobResponse(BaseModel):
    """Response containing job information."""

    id: str
    job_type: str
    source: str
    status: str
    library_id: str | None = None
    library_name: str | None = None
    collection: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    documents_count: int = 0
    chunks_count: int = 0


class ChunkResponse(BaseModel):
    """Response containing a chunk."""

    id: str
    document_id: str
    source: str
    source_type: str
    title: str
    description: str
    content: str
    code_blocks: list[str]
    language: str | None
    score: float


class SearchResponse(BaseModel):
    """Response containing search results."""

    query: str
    results: list[ChunkResponse]
    total: int


class StatsResponse(BaseModel):
    """Response containing collection statistics."""

    collection_name: str
    indexed_vectors_count: int
    points_count: int
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    qdrant_connected: bool


# =============================================================================
# Configuration
# =============================================================================


def get_config() -> dict[str, Any]:
    """Get service configuration from environment."""
    return {
        "qdrant_host": os.environ.get("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.environ.get("QDRANT_PORT", "6333")),
        "default_collection": os.environ.get("DEFAULT_COLLECTION", "deepcontext_chunks"),
    }


# =============================================================================
# Background Worker
# =============================================================================


class BackgroundWorkerManager:
    """Manages the background worker task."""

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_worker(self) -> None:
        """Run the worker loop."""
        queue = JobQueue()
        worker = JobWorker(queue=queue, log_callback=lambda msg: print(f"[worker] {msg}"))

        while self._running:
            try:
                # Run synchronous job processing in thread pool
                job = queue.get_pending_job()
                if job:
                    await asyncio.to_thread(worker.process_job, job)
                else:
                    await asyncio.sleep(2.0)
            except Exception as e:
                print(f"[worker] Error: {e}")
                await asyncio.sleep(2.0)


worker_manager = BackgroundWorkerManager()


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting DeepContext service...")
    await worker_manager.start()
    print("Background worker started")
    
    # Start MCP session manager for streamable-http transport
    async with mcp_server.session_manager.run():
        print("MCP server started at /mcp")
        yield

    # Shutdown
    print("Shutting down...")
    await worker_manager.stop()
    print("Background worker stopped")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="DeepContext",
    description="Semantic chunking and search service for documents, repos, and webpages",
    version="0.1.0",
    lifespan=lifespan,
    routes=[
        # Mount MCP server at /mcp for streamable-http transport
        Mount("/mcp", app=mcp_server.streamable_http_app()),
    ],
)

# Add CORS middleware for browser-based MCP clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],  # Required for MCP session management
)


# =============================================================================
# Helper Functions
# =============================================================================


def job_to_response(job: IngestJob) -> JobResponse:
    """Convert an IngestJob to a JobResponse."""
    return JobResponse(
        id=job.id,
        job_type=job.job_type.value,
        source=job.source,
        status=job.status.value,
        library_id=job.library_id,
        library_name=job.library_name,
        collection=job.collection,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        documents_count=job.documents_count,
        chunks_count=job.chunks_count,
    )


def result_to_response(result: SearchResult) -> ChunkResponse:
    """Convert a SearchResult to a ChunkResponse."""
    chunk = result.chunk
    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        source=chunk.source,
        source_type=chunk.source_type.value,
        title=chunk.title,
        description=chunk.description,
        content=chunk.content,
        code_blocks=chunk.code_blocks,
        language=chunk.language,
        score=result.score,
    )


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check service health and connectivity."""
    config = get_config()
    qdrant_connected = False

    try:
        store = VectorStore(
            host=config["qdrant_host"],
            port=config["qdrant_port"],
        )
        store.client.get_collections()
        qdrant_connected = True
        store.close()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        version="0.1.0",
        qdrant_connected=qdrant_connected,
    )


# =============================================================================
# Ingest Endpoints
# =============================================================================


@app.post("/ingest", response_model=JobResponse, tags=["Ingest"])
async def ingest_source(request: IngestRequest) -> JobResponse:
    """
    Queue a source for ingestion.

    Accepts URLs (GitHub, Confluence, webpages) or local file paths.
    Returns immediately with a job ID - processing happens in background.
    
    Library ID and collection name are automatically derived from the source.
    Library metadata is stored in the 'libraries' collection.
    """
    config = get_config()

    # Auto-derive library_id and collection
    library_id = derive_library_id(request.source, "single")
    library_name = request.source
    collection = library_id_to_collection_name(library_id)

    job = IngestJob(
        id=generate_job_id(),
        job_type=JobType.SINGLE,
        source=request.source,
        library_id=library_id,
        library_name=library_name,
        collection=collection,
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        chunk_size=request.chunk_size,
        threshold=request.threshold,
        code_aware=request.code_aware,
    )

    queue = JobQueue()
    queue.submit(job)

    return job_to_response(job)


@app.post("/ingest/repo", response_model=JobResponse, tags=["Ingest"])
async def ingest_repository(request: IngestRepoRequest) -> JobResponse:
    """
    Queue a GitHub repository for ingestion.

    Fetches all matching files from the repository and indexes them.
    
    Library ID and collection name are automatically derived from the repo.
    Library metadata is stored in the 'libraries' collection.
    """
    config = get_config()

    # Auto-derive library_id and collection
    library_id = derive_library_id(request.repo, "repo")
    library_name = request.repo.split("/")[-1] if "/" in request.repo else request.repo
    collection = library_id_to_collection_name(library_id)

    job = IngestJob(
        id=generate_job_id(),
        job_type=JobType.REPO,
        source=request.repo,
        library_id=library_id,
        library_name=library_name,
        collection=collection,
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        branch=request.branch,
        path=request.path,
        extensions=request.extensions,
    )

    queue = JobQueue()
    queue.submit(job)

    return job_to_response(job)


@app.post("/ingest/website", response_model=JobResponse, tags=["Ingest"])
async def ingest_website(request: IngestWebsiteRequest) -> JobResponse:
    """
    Queue a website for ingestion.

    Crawls the website using sitemap or link following.
    
    Library ID and collection name are automatically derived from the URL.
    Library metadata is stored in the 'libraries' collection.
    """
    config = get_config()

    # Auto-derive library_id and collection
    library_id = derive_library_id(request.url, "website")
    from urllib.parse import urlparse
    parsed = urlparse(request.url)
    library_name = parsed.netloc.replace("www.", "")
    collection = library_id_to_collection_name(library_id)

    job = IngestJob(
        id=generate_job_id(),
        job_type=JobType.WEBSITE,
        source=request.url,
        library_id=library_id,
        library_name=library_name,
        collection=collection,
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        max_pages=request.max_pages,
        url_pattern=request.url_pattern,
        use_sitemap=request.use_sitemap,
    )

    queue = JobQueue()
    queue.submit(job)

    return job_to_response(job)


@app.post("/ingest/confluence", response_model=JobResponse, tags=["Ingest"])
async def ingest_confluence(request: IngestConfluenceRequest) -> JobResponse:
    """
    Queue a Confluence space for ingestion.

    Requires CONFLUENCE_EMAIL and CONFLUENCE_TOKEN environment variables.
    
    Library ID and collection name are automatically derived from the base URL and space key.
    Library metadata is stored in the 'libraries' collection.
    """
    config = get_config()

    # Auto-derive library_id and collection
    from urllib.parse import urlparse
    parsed = urlparse(request.base_url)
    library_id = f"/{parsed.netloc.replace('www.', '')}/{request.space_key}"
    library_name = request.space_key
    collection = library_id_to_collection_name(library_id)

    job = IngestJob(
        id=generate_job_id(),
        job_type=JobType.CONFLUENCE_SPACE,
        source=request.base_url,
        library_id=library_id,
        library_name=library_name,
        collection=collection,
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        space_key=request.space_key,
        limit=request.limit,
    )

    queue = JobQueue()
    queue.submit(job)

    return job_to_response(job)


# =============================================================================
# Job Endpoints
# =============================================================================


@app.get("/jobs", response_model=list[JobResponse], tags=["Jobs"])
async def list_jobs(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum jobs to return"),
) -> list[JobResponse]:
    """List all jobs in the queue."""
    queue = JobQueue()
    status_filter = JobStatus(status) if status else None
    jobs = queue.list_jobs(status=status_filter, limit=limit)
    return [job_to_response(job) for job in jobs]


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job(job_id: str) -> JobResponse:
    """Get details for a specific job."""
    queue = JobQueue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job_to_response(job)


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str) -> dict[str, str]:
    """Delete/cancel a job."""
    queue = JobQueue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a job that is currently processing",
        )

    queue.delete_job(job_id)
    return {"status": "deleted", "job_id": job_id}


@app.delete("/jobs", tags=["Jobs"])
async def clear_completed_jobs() -> dict[str, int]:
    """Clear all completed and failed jobs."""
    queue = JobQueue()
    count = queue.clear_completed()
    return {"deleted": count}


# =============================================================================
# Search Endpoints
# =============================================================================


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search for semantically similar chunks.

    Returns chunks ranked by similarity to the query.
    """
    config = get_config()
    
    # Convert library_id to collection name
    collection = library_id_to_collection_name(request.library_id)

    store = VectorStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        collection_name=collection,
    )

    try:
        results = await asyncio.to_thread(
            store.search,
            query=request.query,
            limit=request.limit,
            source_type=request.source_type,
            language=request.language,
        )
    finally:
        store.close()

    return SearchResponse(
        query=request.query,
        results=[result_to_response(r) for r in results],
        total=len(results),
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., min_length=1, description="Search query"),
    library_id: str = Query(..., description="Library ID to search (e.g., /better-auth/better-auth)"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum results"),
    source_type: str | None = Query(default=None, description="Filter by source type"),
    language: str | None = Query(default=None, description="Filter by code language"),
) -> SearchResponse:
    """
    Search for semantically similar chunks (GET version).

    Convenience endpoint for simple queries.
    """
    request = SearchRequest(
        query=q,
        library_id=library_id,
        limit=limit,
        source_type=SourceType(source_type) if source_type else None,
        language=language,
    )
    return await search(request)


# =============================================================================
# Stats Endpoints
# =============================================================================


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats(
    library_id: str = Query(..., description="Library ID (e.g., /better-auth/better-auth)"),
) -> StatsResponse:
    """Get statistics for a library's collection."""
    config = get_config()
    
    # Convert library_id to collection name
    collection = library_id_to_collection_name(library_id)

    store = VectorStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        collection_name=collection,
    )

    try:
        stats = await asyncio.to_thread(store.get_stats)
    finally:
        store.close()

    return StatsResponse(
        collection_name=stats["collection_name"],
        indexed_vectors_count=stats["indexed_vectors_count"],
        points_count=stats["points_count"],
        status=stats["status"],
    )


@app.delete("/collections/{library_id:path}", tags=["Stats"])
async def clear_collection(library_id: str) -> dict[str, str]:
    """Clear all data from a library's collection."""
    config = get_config()
    
    # Ensure library_id starts with /
    if not library_id.startswith("/"):
        library_id = "/" + library_id
    
    # Convert library_id to collection name
    collection = library_id_to_collection_name(library_id)

    store = VectorStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        collection_name=collection,
    )

    try:
        await asyncio.to_thread(store.clear)
    finally:
        store.close()

    return {"status": "cleared", "library_id": library_id, "collection": collection}


# =============================================================================
# Library Endpoints
# =============================================================================


class LibraryResponse(BaseModel):
    """Response containing library information."""

    library_id: str
    name: str
    collection_name: str
    source: str
    source_type: str
    updated_at: datetime
    created_at: datetime
    documents_count: int = 0
    chunks_count: int = 0


@app.get("/libraries", response_model=list[LibraryResponse], tags=["Libraries"])
async def list_libraries(
    limit: int = Query(default=100, ge=1, le=500, description="Maximum libraries to return"),
) -> list[LibraryResponse]:
    """List all indexed libraries."""
    config = get_config()

    library_store = LibraryStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
    )

    try:
        libraries = await asyncio.to_thread(library_store.list_libraries, limit)
        return [
            LibraryResponse(
                library_id=lib.library_id,
                name=lib.name,
                collection_name=lib.collection_name,
                source=lib.source,
                source_type=lib.source_type,
                updated_at=lib.updated_at,
                created_at=lib.created_at,
                documents_count=lib.documents_count,
                chunks_count=lib.chunks_count,
            )
            for lib in libraries
        ]
    finally:
        library_store.close()


@app.get("/libraries/{library_id:path}", response_model=LibraryResponse, tags=["Libraries"])
async def get_library(library_id: str) -> LibraryResponse:
    """Get details for a specific library by its ID (e.g., /vercel/next.js)."""
    config = get_config()
    
    # Ensure library_id starts with /
    if not library_id.startswith("/"):
        library_id = "/" + library_id

    library_store = LibraryStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
    )

    try:
        library = await asyncio.to_thread(library_store.get_library, library_id)
        if not library:
            raise HTTPException(status_code=404, detail=f"Library not found: {library_id}")
        
        return LibraryResponse(
            library_id=library.library_id,
            name=library.name,
            collection_name=library.collection_name,
            source=library.source,
            source_type=library.source_type,
            updated_at=library.updated_at,
            created_at=library.created_at,
            documents_count=library.documents_count,
            chunks_count=library.chunks_count,
        )
    finally:
        library_store.close()


@app.delete("/libraries/{library_id:path}", tags=["Libraries"])
async def delete_library(library_id: str) -> dict[str, str]:
    """Delete a library and all its chunks."""
    config = get_config()
    
    # Ensure library_id starts with /
    if not library_id.startswith("/"):
        library_id = "/" + library_id

    library_store = LibraryStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
    )

    try:
        library = await asyncio.to_thread(library_store.get_library, library_id)
        if not library:
            raise HTTPException(status_code=404, detail=f"Library not found: {library_id}")
        
        # Delete the chunks collection
        collection_name = library.collection_name
        store = VectorStore(
            host=config["qdrant_host"],
            port=config["qdrant_port"],
            collection_name=collection_name,
        )
        try:
            await asyncio.to_thread(store.client.delete_collection, collection_name)
        except Exception:
            pass  # Collection might not exist
        finally:
            store.close()
        
        # Delete the library record
        await asyncio.to_thread(library_store.delete_library, library_id)
        
        return {"status": "deleted", "library_id": library_id, "collection": collection_name}
    finally:
        library_store.close()


# =============================================================================
# Run with: uvicorn deepcontext.api:app --reload
# =============================================================================


