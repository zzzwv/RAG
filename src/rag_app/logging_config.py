from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


class SensitiveDataFilter(logging.Filter):
    PATTERNS = (
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
        re.compile(r"(?i)((?:api[_-]?key|password|secret)\s*[=:]\s*)[^\s,;]+"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in self.PATTERNS:
            message = pattern.sub(r"\1***", message)
        record.msg = message
        record.args = ()
        return True


def configure_logging(directory: str | Path = "data/logs") -> logging.Logger:
    log_dir = Path(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rag_app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            log_dir / "rag.log", maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
    return logger


def read_recent_logs(directory: str | Path = "data/logs", limit: int = 200) -> list[str]:
    path = Path(directory, "rag.log")
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
