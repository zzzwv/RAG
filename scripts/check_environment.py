from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULES = [
    "pypdf", "docx", "bs4", "langchain", "langchain_chroma", "chromadb",
    "rank_bm25", "jieba", "sentence_transformers", "streamlit", "requests", "cryptography",
]


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"OS: {platform.platform()}")
    failed = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {name}: {version}")
        except Exception as exc:
            failed.append(name)
            print(f"[FAIL] {name}: {exc}")
    try:
        import torch

        print(f"[OK] torch: {torch.__version__}; CUDA={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        failed.append("torch")
        print(f"[FAIL] torch: {exc}")
    office = shutil.which("soffice") or shutil.which("libreoffice")
    print(f"LibreOffice: {office or '未安装（仅影响旧版 .doc）'}")
    for directory in (ROOT / "data" / "chroma", ROOT / "data" / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Writable: {directory}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
