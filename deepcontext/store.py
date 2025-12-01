"""Vector store using Qdrant for semantic search."""

import hashlib
import re
from datetime import datetime
from typing import Generator

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from deepcontext.models import Chunk, Library, SearchResult, SourceType


def library_id_to_collection_name(library_id: str) -> str:
    """
    Convert a library ID to a valid Qdrant collection name.
    
    Examples:
        /vercel/next.js -> vercel__next_js
        /shadcn-ui/ui -> shadcn-ui__ui
        /org/project/v1.0.0 -> org__project__v1_0_0
    """
    # Remove leading slash
    clean_id = library_id.lstrip("/")
    # Replace / with __
    clean_id = clean_id.replace("/", "__")
    # Replace . with _
    clean_id = clean_id.replace(".", "_")
    # Replace any other invalid chars with _
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", clean_id)
    return clean_id


def derive_library_id(source: str, job_type: str) -> str:
    """
    Derive a library ID from the source and job type.
    
    Returns format like /org/project or /domain/path
    """
    if job_type == "repo":
        # GitHub repo: owner/repo -> /owner/repo
        # Remove any github.com prefix if present
        source = source.replace("https://github.com/", "").replace("http://github.com/", "")
        source = source.rstrip("/")
        if not source.startswith("/"):
            source = "/" + source
        return source
    elif job_type == "website":
        # Website: extract domain
        from urllib.parse import urlparse
        parsed = urlparse(source)
        domain = parsed.netloc.replace("www.", "")
        return f"/{domain}"
    elif job_type == "confluence_space":
        # Confluence: /confluence/space_key
        from urllib.parse import urlparse
        parsed = urlparse(source)
        domain = parsed.netloc.replace("www.", "")
        return f"/{domain}"
    else:
        # Single source - try to extract meaningful ID
        if "github.com" in source:
            source = source.replace("https://github.com/", "").replace("http://github.com/", "")
            parts = source.split("/")[:2]
            return "/" + "/".join(parts)
        return "/" + hashlib.md5(source.encode()).hexdigest()[:12]


class LibraryStore:
    """Store for managing library metadata in Qdrant."""
    
    COLLECTION_NAME = "libraries"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.host = host
        self.port = port
        self._client: QdrantClient | None = None
    
    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client
    
    def ensure_collection(self) -> None:
        """Ensure the libraries collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.COLLECTION_NAME not in collection_names:
            # Libraries collection uses a simple 1-dimensional vector (placeholder)
            # We're not doing semantic search on libraries, just storing metadata
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1,  # Minimal vector, we only care about payload
                    distance=Distance.COSINE,
                ),
            )
    
    def upsert_library(self, library: Library) -> None:
        """Insert or update a library record."""
        self.ensure_collection()
        
        point = PointStruct(
            id=self._library_id_to_int(library.library_id),
            vector=[0.0],  # Placeholder vector
            payload={
                "library_id": library.library_id,
                "name": library.name,
                "collection_name": library.collection_name,
                "source": library.source,
                "source_type": library.source_type,
                "updated_at": library.updated_at.isoformat(),
                "created_at": library.created_at.isoformat(),
                "documents_count": library.documents_count,
                "chunks_count": library.chunks_count,
                "metadata": library.metadata,
            },
        )
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )
    
    def get_library(self, library_id: str) -> Library | None:
        """Get a library by its ID."""
        try:
            results = self.client.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[self._library_id_to_int(library_id)],
            )
            if results:
                payload = results[0].payload or {}
                return Library(
                    library_id=payload.get("library_id", ""),
                    name=payload.get("name", ""),
                    collection_name=payload.get("collection_name", ""),
                    source=payload.get("source", ""),
                    source_type=payload.get("source_type", ""),
                    updated_at=datetime.fromisoformat(payload["updated_at"]) if payload.get("updated_at") else datetime.now(),
                    created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.now(),
                    documents_count=payload.get("documents_count", 0),
                    chunks_count=payload.get("chunks_count", 0),
                    metadata=payload.get("metadata", {}),
                )
        except Exception:
            pass
        return None
    
    def list_libraries(self, limit: int = 100) -> list[Library]:
        """List all libraries."""
        try:
            self.ensure_collection()
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=limit,
            )
            libraries = []
            for point in results[0]:
                payload = point.payload or {}
                libraries.append(Library(
                    library_id=payload.get("library_id", ""),
                    name=payload.get("name", ""),
                    collection_name=payload.get("collection_name", ""),
                    source=payload.get("source", ""),
                    source_type=payload.get("source_type", ""),
                    updated_at=datetime.fromisoformat(payload["updated_at"]) if payload.get("updated_at") else datetime.now(),
                    created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.now(),
                    documents_count=payload.get("documents_count", 0),
                    chunks_count=payload.get("chunks_count", 0),
                    metadata=payload.get("metadata", {}),
                ))
            return libraries
        except Exception:
            return []
    
    def delete_library(self, library_id: str) -> bool:
        """Delete a library record."""
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[self._library_id_to_int(library_id)],
            )
            return True
        except Exception:
            return False
    
    def _library_id_to_int(self, library_id: str) -> int:
        """Convert a library ID string to an integer for Qdrant."""
        return int(hashlib.sha256(library_id.encode()).hexdigest()[:16], 16)
    
    def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None


class VectorStore:
    """Vector store using Qdrant for semantic chunk storage and retrieval."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "deepcontext_chunks",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        self._client: QdrantClient | None = None
        self._embedding_model: TextEmbedding | None = None
        self._vector_size: int | None = None

    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            # Use remote Qdrant server
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    @property
    def embedding_model(self) -> TextEmbedding:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(model_name=self.embedding_model_name)
        return self._embedding_model

    @property
    def vector_size(self) -> int:
        """Get the embedding vector size."""
        if self._vector_size is None:
            # Get size by embedding a test string
            test_embedding = list(self.embedding_model.embed(["test"]))[0]
            self._vector_size = len(test_embedding)
        return self._vector_size

    def ensure_collection(self) -> None:
        """Ensure the collection exists with proper configuration."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def index_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
    ) -> int:
        """
        Index chunks into the vector store.

        Returns the number of chunks indexed.
        """
        self.ensure_collection()

        points: list[PointStruct] = []
        indexed_count = 0

        for chunk in chunks:
            # Generate embedding for the chunk
            # Combine title, description, and content for better semantic representation
            text_to_embed = f"{chunk.title}\n{chunk.description}\n{chunk.content}"
            embedding = list(self.embedding_model.embed([text_to_embed]))[0]

            # Create point with payload
            point = PointStruct(
                id=self._chunk_id_to_int(chunk.id),
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "source_type": chunk.source_type.value,
                    "title": chunk.title,
                    "description": chunk.description,
                    "content": chunk.content,
                    "code_blocks": chunk.code_blocks,
                    "language": chunk.language,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                },
            )
            points.append(point)

            # Batch upsert
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                indexed_count += len(points)
                points = []

        # Upsert remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            indexed_count += len(points)

        return indexed_count

    def search(
        self,
        query: str,
        limit: int = 10,
        source_type: SourceType | None = None,
        language: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for semantically similar chunks.

        Args:
            query: The search query
            limit: Maximum number of results to return
            source_type: Filter by source type
            language: Filter by code language

        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = list(self.embedding_model.embed([query]))[0]

        # Build filter if needed
        query_filter = None
        if source_type or language:
            must_conditions = []
            if source_type:
                must_conditions.append(
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type.value),
                    )
                )
            if language:
                must_conditions.append(
                    FieldCondition(
                        key="language",
                        match=MatchValue(value=language.lower()),
                    )
                )
            query_filter = Filter(must=must_conditions)

        # Search using query_points (new API)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            query_filter=query_filter,
        )

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []
        for result in results.points:
            payload = result.payload or {}
            chunk = Chunk(
                id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                source=payload.get("source", ""),
                source_type=SourceType(payload.get("source_type", "markdown")),
                title=payload.get("title", ""),
                description=payload.get("description", ""),
                content=payload.get("content", ""),
                code_blocks=payload.get("code_blocks", []),
                language=payload.get("language"),
                token_count=payload.get("token_count", 0),
                metadata=payload.get("metadata", {}),
            )
            search_results.append(SearchResult(chunk=chunk, score=result.score))

        return search_results

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source.

        Returns the number of chunks deleted.
        """
        # Get count before deletion
        count_before = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        ).count

        # Delete points matching the source
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )

        return count_before

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks from a specific document.

        Returns the number of chunks deleted.
        """
        # Get count before deletion
        count_before = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id))
                ]
            ),
        ).count

        # Delete points matching the document_id
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id))
                ]
            ),
        )

        return count_before

    def get_stats(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "indexed_vectors_count": 0,
                "points_count": 0,
                "status": f"not_found ({e})",
            }

    def clear(self) -> None:
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection might not exist
        self.ensure_collection()

    def _chunk_id_to_int(self, chunk_id: str) -> int:
        """Convert a string chunk ID to an integer for Qdrant."""
        # Use hash to convert string ID to a large integer
        return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:16], 16)

    def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

