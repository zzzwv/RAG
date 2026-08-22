from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from rag_app.chat.llm import OpenAICompatibleLLM
from rag_app.chat.service import QAService
from rag_app.chunking.service import ChunkingService
from rag_app.config import AppSettings
from rag_app.indexing.bm25_store import BM25Store
from rag_app.indexing.chroma_store import BGEEmbeddings, ChromaStore
from rag_app.indexing.service import IngestionService
from rag_app.logging_config import configure_logging
from rag_app.parsing.pipeline import DocumentParsingPipeline
from rag_app.retrieval.hybrid import HybridRetriever
from rag_app.retrieval.query_processor import RetrievalQueryProcessor
from rag_app.retrieval.reranker import BGEReranker
from rag_app.security.secrets import EncryptedSecretStore


def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_api_key() -> str:
    if os.getenv("RAG_MASTER_KEY"):
        try:
            return EncryptedSecretStore().get("api_key", "not-required")
        except Exception:
            logging.getLogger("rag_app").exception("读取加密 API 密钥失败")
    return os.getenv("RAG_API_KEY", "not-required")


@dataclass(slots=True)
class AppRuntime:
    settings: AppSettings
    ingestion: IngestionService
    retriever: HybridRetriever
    reranker: BGEReranker
    llm: OpenAICompatibleLLM
    vector_store: ChromaStore
    logger: logging.Logger

    def create_qa(self, memory) -> QAService:
        return QAService(
            retriever=self.retriever,
            reranker=self.reranker,
            llm=self.llm,
            memory=memory,
            query_processor=RetrievalQueryProcessor(self.settings.retrieval_query_max_length),
            max_question_length=self.settings.max_question_length,
        )


def build_runtime(settings_path: str = "config/settings.json") -> AppRuntime:
    settings = AppSettings.load(settings_path)
    logger = configure_logging(settings.log_directory)
    device = detect_device()
    embeddings = BGEEmbeddings(settings.embedding_model, device=device)
    vector_store = ChromaStore(settings.persist_directory, embedding_function=embeddings)
    bm25_store = BM25Store()
    bm25_store.rebuild(vector_store.all_chunks())
    parser = DocumentParsingPipeline(max_file_size=settings.max_file_size)
    chunker = ChunkingService(
        normal_size=settings.normal_chunk_size,
        normal_overlap=settings.normal_chunk_overlap,
        technical_size=settings.technical_chunk_size,
        technical_overlap=settings.technical_chunk_overlap,
    )
    ingestion = IngestionService(parser, chunker, vector_store, bm25_store)
    retriever = HybridRetriever(
        vector_store,
        bm25_store,
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        fused_limit=settings.fused_top_k,
        rrf_k=settings.rrf_k,
    )
    reranker = BGEReranker(
        settings.reranker_model,
        threshold=settings.rerank_threshold,
        top_n=settings.rerank_top_n,
        device=device,
    )
    llm = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=load_api_key(),
        timeout=settings.llm_timeout,
        max_retries=settings.llm_retries,
    )
    logger.info("应用资源初始化完成 device=%s chunks=%s", device, vector_store.count())
    return AppRuntime(settings, ingestion, retriever, reranker, llm, vector_store, logger)
