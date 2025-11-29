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

