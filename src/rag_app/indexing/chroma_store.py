from __future__ import annotations

import threading
from pathlib import Path

from rag_app.models import ChunkRecord, SearchHit


class BGEEmbeddings:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh",
        device: str | None = None,
        *,
        model=None,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.query_instruction = query_instruction
        self._model = model

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([f"{self.query_instruction}{text}"])[0]


class ChromaStore:
    def __init__(
        self,
        persist_directory: str | Path,
        *,
        collection_name: str = "enterprise_knowledge",
        embedding_function=None,
        vector_store=None,
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.embedding_function = embedding_function or BGEEmbeddings()
        if vector_store is None:
            from langchain_chroma import Chroma

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_metadata={"hnsw:space": "cosine"},
            )
        self.vector_store = vector_store
        self._lock = threading.RLock()

    @staticmethod
    def _from_raw(chunk_id: str, content: str, metadata: dict) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=chunk_id,
            source=str(metadata.get("doc_source", "unknown")),
            doc_type=str(metadata.get("doc_type", "unknown")),
            content=content,
            create_time=str(metadata.get("create_time", "")),
            location=str(metadata.get("location", "body")),
            content_hash=str(metadata.get("content_hash", "")),
            chunk_profile=str(metadata.get("chunk_profile", "normal")),
            index=int(metadata.get("chunk_index", 0)),
        )

    def replace_source(self, source: str, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        with self._lock:
            existing = self.vector_store.get(where={"doc_source": source}, include=[])
            old_ids = set(existing.get("ids", []))
            new_ids = {chunk.chunk_id for chunk in chunks}
            self.vector_store.add_texts(
                texts=[chunk.content for chunk in chunks],
                metadatas=[chunk.metadata() for chunk in chunks],
                ids=[chunk.chunk_id for chunk in chunks],
            )
            stale_ids = sorted(old_ids - new_ids)
            if stale_ids:
                self.vector_store.delete(ids=stale_ids)

    def all_chunks(self) -> list[ChunkRecord]:
        raw = self.vector_store.get(include=["documents", "metadatas"])
        return [
            self._from_raw(chunk_id, content, metadata or {})
            for chunk_id, content, metadata in zip(
                raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])
            )
        ]

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        if self.count() == 0:
            return []
        documents = self.vector_store.similarity_search_with_score(query, k=top_k)
        results: list[SearchHit] = []
        for document, distance in documents:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if not chunk_id:
                chunk_id = str(document.id or document.metadata.get("content_hash", ""))
            chunk = self._from_raw(chunk_id, document.page_content, document.metadata)
            results.append(SearchHit(chunk=chunk, vector_score=1.0 / (1.0 + float(distance))))
        return results

    def count(self) -> int:
        return int(self.vector_store._collection.count())

    def clear(self) -> None:
        with self._lock:
            raw = self.vector_store.get(include=[])
            ids = raw.get("ids", [])
            if ids:
                self.vector_store.delete(ids=ids)

    def close(self) -> None:
        """Release Chroma's persistent file handles (important on Windows)."""
        client = getattr(self.vector_store, "_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
