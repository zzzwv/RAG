from __future__ import annotations

from collections import deque

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:  # Lets diagnostics/tests run before optional dependencies are installed.
    class BaseMessage:  # type: ignore[no-redef]
        def __init__(self, content: str) -> None:
            self.content = content

    class HumanMessage(BaseMessage):  # type: ignore[no-redef]
        pass

    class AIMessage(BaseMessage):  # type: ignore[no-redef]
        pass


class WindowMemory:
    def __init__(self, max_turns: int = 10, max_chars: int = 12_000) -> None:
        if max_turns < 1 or max_chars < 1:
            raise ValueError("max_turns and max_chars must be positive")
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add_turn(self, user_query: str, answer: str) -> None:
        self._turns.append((user_query, answer))

    def messages(self) -> list[BaseMessage]:
        selected: list[tuple[str, str]] = []
        used = 0
        for user, assistant in reversed(self._turns):
            turn_size = len(user) + len(assistant)
            if used + turn_size > self.max_chars:
                break
            selected.append((user, assistant))
            used += turn_size
        messages: list[BaseMessage] = []
        for user, assistant in reversed(selected):
            messages.extend((HumanMessage(content=user), AIMessage(content=assistant)))
        return messages

    def user_queries(self) -> list[str]:
        return [user for user, _ in self._turns]

    def clear(self) -> None:
        self._turns.clear()
