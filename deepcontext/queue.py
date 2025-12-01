"""Job queue for asynchronous ingestion using SQLite."""

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from deepcontext.chunker import ChunkConfig, MarkdownCodeChunker, SemanticChunker
from deepcontext.fetcher import ContentFetcher
from deepcontext.models import IngestJob, JobStatus, JobType, Library
from deepcontext.store import LibraryStore, VectorStore


def get_db_path() -> Path:
    """Get the path to the SQLite database."""
    # Store in user's data directory
    data_dir = Path.home() / ".deepcontext"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "jobs.db"


class JobQueue:
    """SQLite-backed job queue for ingestion tasks."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_db_path()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    library_id TEXT,
                    library_name TEXT,
                    collection TEXT NOT NULL DEFAULT 'deepcontext_chunks',
                    host TEXT NOT NULL DEFAULT 'localhost',
                    port INTEGER NOT NULL DEFAULT 6333,
                    chunk_size INTEGER NOT NULL DEFAULT 1024,
                    threshold REAL NOT NULL DEFAULT 0.7,
                    code_aware INTEGER NOT NULL DEFAULT 1,
                    branch TEXT,
                    path TEXT,
                    extensions TEXT,
                    max_pages INTEGER DEFAULT 100,
                    url_pattern TEXT,
                    use_sitemap INTEGER DEFAULT 1,
                    space_key TEXT,
                    job_limit INTEGER DEFAULT 100,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    documents_count INTEGER DEFAULT 0,
                    chunks_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)
            """)
            # Add new columns if they don't exist (for migration)
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN library_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN library_name TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    def submit(self, job: IngestJob) -> str:
        """Submit a job to the queue. Returns the job ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, job_type, source, status, library_id, library_name,
                    collection, host, port, chunk_size, threshold, code_aware,
                    branch, path, extensions, max_pages, url_pattern, use_sitemap,
                    space_key, job_limit, created_at, documents_count, chunks_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.job_type.value,
                    job.source,
                    job.status.value,
                    job.library_id,
                    job.library_name,
                    job.collection,
                    job.host,
                    job.port,
                    job.chunk_size,
                    job.threshold,
                    1 if job.code_aware else 0,
                    job.branch,
                    job.path,
                    job.extensions,
                    job.max_pages,
                    job.url_pattern,
                    1 if job.use_sitemap else 0,
                    job.space_key,
                    job.limit,
                    job.created_at.isoformat(),
                    job.documents_count,
                    job.chunks_count,
                ),
            )
            conn.commit()
        return job.id

    def get_job(self, job_id: str) -> IngestJob | None:
        """Get a job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_job(row)
        return None

    def get_pending_job(self) -> IngestJob | None:
        """Get the next pending job (FIFO)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
                (JobStatus.PENDING.value,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_job(row)
        return None

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
        documents_count: int | None = None,
        chunks_count: int | None = None,
    ) -> None:
        """Update job status and related fields."""
        with sqlite3.connect(self.db_path) as conn:
            updates = ["status = ?"]
            params: list = [status.value]

            if status == JobStatus.PROCESSING:
                updates.append("started_at = ?")
                params.append(datetime.now().isoformat())
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                updates.append("completed_at = ?")
                params.append(datetime.now().isoformat())

            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

            if documents_count is not None:
                updates.append("documents_count = ?")
                params.append(documents_count)

            if chunks_count is not None:
                updates.append("chunks_count = ?")
                params.append(chunks_count)

            params.append(job_id)
            conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[IngestJob]:
        """List jobs, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                cursor = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_job(row) for row in cursor.fetchall()]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_completed(self) -> int:
        """Clear all completed and failed jobs. Returns count deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE status IN (?, ?)",
                (JobStatus.COMPLETED.value, JobStatus.FAILED.value),
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_job(self, row: sqlite3.Row) -> IngestJob:
        """Convert a database row to an IngestJob."""
        return IngestJob(
            id=row["id"],
            job_type=JobType(row["job_type"]),
            source=row["source"],
            status=JobStatus(row["status"]),
            library_id=row["library_id"] if "library_id" in row.keys() else None,
            library_name=row["library_name"] if "library_name" in row.keys() else None,
            collection=row["collection"],
            host=row["host"],
            port=row["port"],
            chunk_size=row["chunk_size"],
            threshold=row["threshold"],
            code_aware=bool(row["code_aware"]),
            branch=row["branch"],
            path=row["path"],
            extensions=row["extensions"],
            max_pages=row["max_pages"],
            url_pattern=row["url_pattern"],
            use_sitemap=bool(row["use_sitemap"]),
            space_key=row["space_key"],
            limit=row["job_limit"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            error_message=row["error_message"],
            documents_count=row["documents_count"],
            chunks_count=row["chunks_count"],
        )


def generate_job_id() -> str:
    """Generate a short, memorable job ID."""
    return uuid.uuid4().hex[:8]


class JobWorker:
    """Background worker that processes ingestion jobs."""

    def __init__(
        self,
        queue: JobQueue | None = None,
        log_callback: Callable[[str], None] | None = None,
    ):
        self.queue = queue or JobQueue()
        self.log = log_callback or print
        self._running = False

    def process_job(self, job: IngestJob) -> None:
        """Process a single ingestion job."""
        self.log(f"[{job.id}] Starting: {job.source}")
        self.queue.update_status(job.id, JobStatus.PROCESSING)

        try:
            if job.job_type == JobType.SINGLE:
                self._process_single(job)
            elif job.job_type == JobType.REPO:
                self._process_repo(job)
            elif job.job_type == JobType.CONFLUENCE_SPACE:
                self._process_confluence_space(job)
            elif job.job_type == JobType.WEBSITE:
                self._process_website(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

        except Exception as e:
            self.log(f"[{job.id}] Failed: {e}")
            self.queue.update_status(job.id, JobStatus.FAILED, error_message=str(e))
            raise

    def _process_single(self, job: IngestJob) -> None:
        """Process a single source ingestion."""
        with ContentFetcher() as fetcher:
            document = fetcher.fetch(job.source)

        self.log(f"[{job.id}] Fetched: {document.title}")

        config = ChunkConfig(max_chunk_size=job.chunk_size, min_chunk_size=job.threshold)
        chunker = MarkdownCodeChunker(config) if job.code_aware else SemanticChunker(config)
        chunks = chunker.chunk_document(document)

        self.log(f"[{job.id}] Created {len(chunks)} chunks")

        store = VectorStore(host=job.host, port=job.port, collection_name=job.collection)
        try:
            indexed = store.index_chunks(chunks)
            self.log(f"[{job.id}] Indexed {indexed} chunks")
        finally:
            store.close()

        # Update library metadata
        if job.library_id:
            self._upsert_library(job, documents_count=1, chunks_count=len(chunks))

        self.queue.update_status(
            job.id,
            JobStatus.COMPLETED,
            documents_count=1,
            chunks_count=len(chunks),
        )
        self.log(f"[{job.id}] Completed")

    def _process_repo(self, job: IngestJob) -> None:
        """Process a GitHub repository ingestion."""
        ext_list = [e.strip() for e in (job.extensions or ".md,.mdx").split(",")]

        with ContentFetcher() as fetcher:
            documents = fetcher.fetch_github_repo(
                repo=job.source,
                branch=job.branch,
                path=job.path or "",
                extensions=ext_list,
            )

        self.log(f"[{job.id}] Fetched {len(documents)} documents")

        config = ChunkConfig(max_chunk_size=job.chunk_size, min_chunk_size=job.threshold)
        chunker = MarkdownCodeChunker(config)

        all_chunks = []
        for doc in documents:
            try:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                self.log(f"[{job.id}] Warning: Error chunking {doc.title}: {e}")

        self.log(f"[{job.id}] Created {len(all_chunks)} chunks")

        store = VectorStore(host=job.host, port=job.port, collection_name=job.collection)
        try:
            indexed = store.index_chunks(all_chunks)
            self.log(f"[{job.id}] Indexed {indexed} chunks")
        finally:
            store.close()

        # Update library metadata
        if job.library_id:
            self._upsert_library(job, documents_count=len(documents), chunks_count=len(all_chunks))

        self.queue.update_status(
            job.id,
            JobStatus.COMPLETED,
            documents_count=len(documents),
            chunks_count=len(all_chunks),
        )
        self.log(f"[{job.id}] Completed")

    def _process_confluence_space(self, job: IngestJob) -> None:
        """Process a Confluence space ingestion."""
        if not job.space_key:
            raise ValueError("space_key is required for Confluence space ingestion")

        with ContentFetcher() as fetcher:
            documents = fetcher.fetch_confluence_space(
                base_url=job.source,
                space_key=job.space_key,
                limit=job.limit,
            )

        self.log(f"[{job.id}] Fetched {len(documents)} pages")

        if not documents:
            if job.library_id:
                self._upsert_library(job, documents_count=0, chunks_count=0)
            self.queue.update_status(job.id, JobStatus.COMPLETED, documents_count=0, chunks_count=0)
            return

        config = ChunkConfig(max_chunk_size=job.chunk_size, min_chunk_size=job.threshold)
        chunker = MarkdownCodeChunker(config)

        all_chunks = []
        for doc in documents:
            try:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                self.log(f"[{job.id}] Warning: Error chunking {doc.title}: {e}")

        self.log(f"[{job.id}] Created {len(all_chunks)} chunks")

        store = VectorStore(host=job.host, port=job.port, collection_name=job.collection)
        try:
            indexed = store.index_chunks(all_chunks)
            self.log(f"[{job.id}] Indexed {indexed} chunks")
        finally:
            store.close()

        # Update library metadata
        if job.library_id:
            self._upsert_library(job, documents_count=len(documents), chunks_count=len(all_chunks))

        self.queue.update_status(
            job.id,
            JobStatus.COMPLETED,
            documents_count=len(documents),
            chunks_count=len(all_chunks),
        )
        self.log(f"[{job.id}] Completed")

    def _process_website(self, job: IngestJob) -> None:
        """Process a website ingestion."""
        with ContentFetcher() as fetcher:
            documents = fetcher.fetch_website(
                base_url=job.source,
                max_pages=job.max_pages,
                url_pattern=job.url_pattern,
                use_sitemap=job.use_sitemap,
            )

        self.log(f"[{job.id}] Fetched {len(documents)} pages")

        if not documents:
            if job.library_id:
                self._upsert_library(job, documents_count=0, chunks_count=0)
            self.queue.update_status(job.id, JobStatus.COMPLETED, documents_count=0, chunks_count=0)
            return

        config = ChunkConfig(max_chunk_size=job.chunk_size, min_chunk_size=job.threshold)
        chunker = MarkdownCodeChunker(config)

        all_chunks = []
        for doc in documents:
            try:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                self.log(f"[{job.id}] Warning: Error chunking {doc.title}: {e}")

        self.log(f"[{job.id}] Created {len(all_chunks)} chunks")

        store = VectorStore(host=job.host, port=job.port, collection_name=job.collection)
        try:
            indexed = store.index_chunks(all_chunks)
            self.log(f"[{job.id}] Indexed {indexed} chunks")
        finally:
            store.close()

        # Update library metadata
        if job.library_id:
            self._upsert_library(job, documents_count=len(documents), chunks_count=len(all_chunks))

        self.queue.update_status(
            job.id,
            JobStatus.COMPLETED,
            documents_count=len(documents),
            chunks_count=len(all_chunks),
        )
        self.log(f"[{job.id}] Completed")

    def _upsert_library(self, job: IngestJob, documents_count: int, chunks_count: int) -> None:
        """Update or create library metadata in the libraries collection."""
        if not job.library_id:
            return

        library_store = LibraryStore(host=job.host, port=job.port)
        try:
            # Check if library already exists to preserve created_at
            existing = library_store.get_library(job.library_id)
            
            library = Library(
                library_id=job.library_id,
                name=job.library_name or job.source,
                collection_name=job.collection,
                source=job.source,
                source_type=job.job_type.value,
                updated_at=datetime.now(),
                created_at=existing.created_at if existing else datetime.now(),
                documents_count=documents_count,
                chunks_count=chunks_count,
            )
            
            library_store.upsert_library(library)
            self.log(f"[{job.id}] Updated library: {job.library_id}")
        finally:
            library_store.close()

    def run_once(self) -> bool:
        """Process one job from the queue. Returns True if a job was processed."""
        job = self.queue.get_pending_job()
        if job:
            self.process_job(job)
            return True
        return False

    def run(self, poll_interval: float = 2.0) -> None:
        """Run the worker continuously, polling for jobs."""
        self._running = True
        self.log("Worker started, polling for jobs...")

        while self._running:
            try:
                if not self.run_once():
                    time.sleep(poll_interval)
            except KeyboardInterrupt:
                self.log("Worker interrupted")
                break
            except Exception as e:
                self.log(f"Worker error: {e}")
                time.sleep(poll_interval)

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False

