"""Vector store using Qdrant for semantic search."""

import hashlib
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

from deepcontext.models import Chunk, SearchResult, SourceType


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

