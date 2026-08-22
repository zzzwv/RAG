from __future__ import annotations

from rag_app.chat.prompts import SYSTEM_PROMPT
from rag_app.exceptions import LLMServiceError


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "not-required",
        timeout: float = 30.0,
        max_retries: int = 2,
        client=None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "not-required"
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client

    def _load(self):
        if self._client is None:
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                base_url=self.base_url,
                model=self.model,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                temperature=0.1,
            )
        return self._client

    def generate(self, *, current_query: str, context: str, history: list) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [SystemMessage(content=SYSTEM_PROMPT), *history]
            messages.append(HumanMessage(content=f"参考资料：\n{context}\n\n用户问题：{current_query}"))
            response = self._load().invoke(messages)
            answer = str(response.content).strip()
            if not answer:
                raise LLMServiceError("问答服务异常，请稍后重试")
            return answer
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError("问答服务异常，请稍后重试") from exc
