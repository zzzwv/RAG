from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence


class RetrievalQueryProcessor:
    """Build a lightweight retrieval query without invoking an LLM."""

    PRONOUNS = ("它", "那个", "这些", "那些", "这个", "那", "上面", "刚才")
    ELLIPSIS_RE = re.compile(
        r"^(?:怎么|如何|为什么|是否|能否|可以|需要|支持|包括|属于|有|是|会|要|多久|多少|什么时候|在哪里|怎么办)"
    )

    def __init__(self, max_length: int = 512) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = max_length

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(unicodedata.normalize("NFC", str(query)).split())

    def _needs_context(self, query: str) -> bool:
        return any(token in query for token in self.PRONOUNS) or (
            len(query) <= 30 and bool(self.ELLIPSIS_RE.match(query))
        )

    def process(self, current_query: str, user_history: Sequence[str] | None = None) -> str:
        current = self._normalize(current_query)
        if not current:
            return current
        if not self._needs_context(current) or not user_history:
            return current[: self.max_length]
        try:
            history = self._normalize(user_history[-1])
        except (IndexError, TypeError, AttributeError):
            return current[: self.max_length]
        if not history:
            return current[: self.max_length]
        remaining = self.max_length - len(current) - 1
        if remaining <= 0:
            return current[: self.max_length]
        return f"{history[:remaining]} {current}"
