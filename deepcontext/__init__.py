"""DeepContext - Semantic chunking service for documents, GitHub repos, and webpages."""

from deepcontext.chunker import SemanticChunker
from deepcontext.fetcher import ContentFetcher
from deepcontext.store import VectorStore
from deepcontext.models import Chunk, Document

__all__ = ["SemanticChunker", "ContentFetcher", "VectorStore", "Chunk", "Document"]
__version__ = "0.1.0"

