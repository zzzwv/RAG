from __future__ import annotations

import re
from enum import Enum


class ChunkProfile(str, Enum):
    NORMAL = "normal"
    TECHNICAL = "technical"


_TECH_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"\b(?:def|class|import|from|return|SELECT|INSERT|UPDATE|function|const|public)\b", re.I),
    re.compile(r"(?:=>|==|!=|\{\s*$|;\s*$)", re.M),
    re.compile(r"(?:API|SDK|HTTP|JSON|SQL|Python|Java|错误码|接口参数)", re.I),
)


def classify_document(text: str, doc_type: str) -> ChunkProfile:
    sample = text[:20_000]
    score = sum(1 for pattern in _TECH_PATTERNS if pattern.search(sample))
    code_lines = sum(1 for line in sample.splitlines() if re.match(r"^\s{2,}\S", line))
    if score >= 2 or "```" in sample or code_lines >= 3:
        return ChunkProfile.TECHNICAL
    return ChunkProfile.NORMAL
