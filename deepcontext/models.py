"""Data models for DeepContext."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of content source."""

    MARKDOWN = "markdown"
    GITHUB = "github"
    WEBPAGE = "webpage"
    CONFLUENCE = "confluence"


class JobStatus(str, Enum):
    """Status of an ingestion job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    """Type of ingestion job."""

    SINGLE = "single"  # Single source ingestion
    REPO = "repo"  # GitHub repository
    CONFLUENCE_SPACE = "confluence_space"  # Confluence space
    WEBSITE = "website"  # Website crawl


class IngestJob(BaseModel):
    """Represents a queued ingestion job."""

    id: str
    job_type: JobType
    source: str
    status: JobStatus = JobStatus.PENDING
    
    # Library identification
    library_id: str | None = None  # e.g., /vercel/next.js - auto-derived if not provided
    library_name: str | None = None  # Human-readable name - defaults to source
    
    # Common options (collection is now auto-generated from library_id)
    collection: str = "deepcontext_chunks"  # Will be overridden based on library_id
    host: str = "localhost"
    port: int = 6333
    chunk_size: int = 1024
    threshold: float = 0.7
    code_aware: bool = True
    
    # Repo-specific options
    branch: str | None = None
    path: str | None = None
    extensions: str | None = None
    
    # Website-specific options
    max_pages: int = 100
    url_pattern: str | None = None
    use_sitemap: bool = True
    
    # Confluence-specific options
    space_key: str | None = None
    limit: int = 100
    
    # Job tracking
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    
    # Results
    documents_count: int = 0
    chunks_count: int = 0


class Document(BaseModel):
    """Represents a fetched document."""

    id: str
    source: str
    source_type: SourceType
    title: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=datetime.now)


class Chunk(BaseModel):
    """Represents a semantic chunk of content."""

    id: str
    document_id: str
    source: str
    source_type: SourceType
    title: str
    description: str
    content: str
    code_blocks: list[str] = Field(default_factory=list)
    language: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_display_format(self) -> str:
        """Format chunk for display, similar to Context7 output."""
        lines = [
            f"### {self.title}",
            "",
            f"Source: {self.source}",
            "",
            self.description,
        ]

        for code_block in self.code_blocks:
            lang = self.language or ""
            lines.extend(["", f"```{lang}", code_block, "```"])

        return "\n".join(lines)


class SearchResult(BaseModel):
    """Represents a search result."""

    chunk: Chunk
    score: float


class Library(BaseModel):
    """Represents a library in the libraries collection."""

    library_id: str  # e.g., /vercel/next.js or /shadcn-ui/ui
    name: str  # Human-readable name
    collection_name: str  # Qdrant collection name for chunks
    source: str  # Original source URL/path
    source_type: str  # repo, website, confluence, etc.
    updated_at: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)
    documents_count: int = 0
    chunks_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

