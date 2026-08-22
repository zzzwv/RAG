from __future__ import annotations

import threading
from dataclasses import dataclass

from rag_app.chunking.classifier import ChunkProfile


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: str
    chunk_count: int
    status: str = "success"


class IngestionService:
    def __init__(self, parser, chunker, vector_store, bm25_store) -> None:
        self.parser = parser
        self.chunker = chunker
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self._lock = threading.RLock()

    def refresh_bm25(self) -> None:
        self.bm25_store.rebuild(self.vector_store.all_chunks())

    def ingest_file(self, filename: str, data: bytes, profile: ChunkProfile | str | None = None) -> IngestResult:
        document = self.parser.parse_file(filename, data)
        chunks = self.chunker.split(document, profile=profile)
        with self._lock:
            self.vector_store.replace_source(document.source, chunks)
            self.refresh_bm25()
        return IngestResult(document.source, len(chunks))

    def ingest_url(self, url: str, profile: ChunkProfile | str | None = None) -> IngestResult:
        document = self.parser.parse_url(url)
        chunks = self.chunker.split(document, profile=profile)
        with self._lock:
            self.vector_store.replace_source(document.source, chunks)
            self.refresh_bm25()
        return IngestResult(document.source, len(chunks))

    def clear(self) -> None:
        with self._lock:
            self.vector_store.clear()
            self.bm25_store.rebuild([])
