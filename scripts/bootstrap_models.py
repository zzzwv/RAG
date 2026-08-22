from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_app.config import AppSettings
from rag_app.runtime import detect_device


def main() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    settings = AppSettings.load(ROOT / "config" / "settings.json")
    device = detect_device()
    print(f"下载/校验向量模型：{settings.embedding_model}")
    SentenceTransformer(settings.embedding_model, device=device)
    print(f"下载/校验重排模型：{settings.reranker_model}")
    CrossEncoder(settings.reranker_model, device=device, max_length=512)
    print(f"模型准备完成，设备：{device}")


if __name__ == "__main__":
    main()
