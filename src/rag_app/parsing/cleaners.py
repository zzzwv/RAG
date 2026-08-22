from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from rag_app.exceptions import InvalidContentError

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"[^\S\r\n]+")
_PAGE_NUMBER_RE = re.compile(r"^(?:第\s*)?\d+\s*(?:页)?(?:\s*/\s*\d+)?$", re.I)
_USEFUL_RE = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)


def _replacement_ratio(text: str) -> float:
    if not text:
        return 1.0
    suspicious = text.count("�") + text.count("\ufffd")
    return suspicious / len(text)


def clean_text(text: str, *, max_replacement_ratio: float = 0.05) -> str:
    if not isinstance(text, str):
        raise InvalidContentError("文档内容无效")
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    if _replacement_ratio(normalized) > max_replacement_ratio:
        raise InvalidContentError("文档乱码比例过高")
    normalized = _CONTROL_RE.sub("", normalized).replace("�", "")
    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = _SPACE_RE.sub(" ", raw_line).strip()
        if not line or not _USEFUL_RE.search(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if not cleaned or len(_USEFUL_RE.findall(cleaned)) < 2:
        raise InvalidContentError("未提取到有效正文")
    return cleaned


def _line_signature(line: str) -> str:
    return re.sub(r"\d+", "#", _SPACE_RE.sub(" ", line).strip().lower())


def remove_repeated_page_lines(pages: list[str], *, frequency: float = 0.6) -> list[str]:
    if len(pages) < 2:
        return pages
    per_page = [set(_line_signature(line) for line in page.splitlines() if line.strip()) for page in pages]
    counts = Counter(signature for signatures in per_page for signature in signatures)
    minimum = max(2, math.ceil(len(pages) * frequency))
    repeated = {key for key, count in counts.items() if count >= minimum and len(key) <= 100}
    result: list[str] = []
    for page in pages:
        kept = []
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped or _PAGE_NUMBER_RE.match(stripped):
                continue
            if _line_signature(stripped) in repeated:
                continue
            kept.append(stripped)
        result.append("\n".join(kept))
    return result
