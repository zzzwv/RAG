from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(slots=True)
class AppSettings:
    persist_directory: str = "data/chroma"
    log_directory: str = "data/logs"
    normal_chunk_size: int = 1000
    normal_chunk_overlap: int = 200
    technical_chunk_size: int = 500
    technical_chunk_overlap: int = 100
    memory_turns: int = 10
    memory_max_chars: int = 12000
    vector_top_k: int = 10
    bm25_top_k: int = 10
    fused_top_k: int = 10
    rrf_k: int = 60
    rerank_top_n: int = 3
    rerank_threshold: float = 0.35
    embedding_model: str = "BAAI/bge-small-zh"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm_timeout: float = 30.0
    llm_retries: int = 2
    max_question_length: int = 500
    retrieval_query_max_length: int = 512
    max_file_size: int = 20 * 1024 * 1024
    admin_password_hash: str = ""

    def validate(self) -> "AppSettings":
        pairs = (
            (self.normal_chunk_size, self.normal_chunk_overlap),
            (self.technical_chunk_size, self.technical_chunk_overlap),
        )
        if any(size < 1 or overlap < 0 or overlap >= size for size, overlap in pairs):
            raise ValueError("切片重叠度必须大于等于 0 且小于切片大小")
        if self.memory_turns < 1 or self.memory_max_chars < 1 or self.rrf_k < 1 or self.rerank_top_n < 1:
            raise ValueError("轮次与检索数量必须为正整数")
        if not 0 <= self.rerank_threshold <= 1:
            raise ValueError("重排阈值必须位于 0 到 1 之间")
        return self

    @classmethod
    def load(cls, path: str | Path = "config/settings.json") -> "AppSettings":
        values: dict = {}
        config_path = Path(path)
        if config_path.exists():
            values.update(json.loads(config_path.read_text(encoding="utf-8")))
        valid_names = {item.name for item in fields(cls)}
        values = {key: value for key, value in values.items() if key in valid_names}
        for name in valid_names:
            env_name = f"RAG_{name.upper()}"
            if env_name not in os.environ:
                continue
            raw = os.environ[env_name]
            field_type = next(item.type for item in fields(cls) if item.name == name)
            if field_type in (int, "int"):
                values[name] = int(raw)
            elif field_type in (float, "float"):
                values[name] = float(raw)
            else:
                values[name] = raw
        return cls(**values).validate()

    def save(self, path: str | Path = "config/settings.json") -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
