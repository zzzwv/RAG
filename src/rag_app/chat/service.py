from __future__ import annotations

import time

from rag_app.chat.prompts import build_context
from rag_app.exceptions import ValidationError
from rag_app.models import AnswerResult

NO_RESULT_MESSAGE = "当前知识库中无相关内容，无法为您解答"


class QAService:
    def __init__(
        self,
        *,
        retriever,
        reranker,
        llm,
        memory,
        query_processor,
        max_question_length: int = 500,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.memory = memory
        self.query_processor = query_processor
        self.max_question_length = max_question_length

    def answer(self, question: str) -> AnswerResult:
        current = question.strip()
        if not current:
            raise ValidationError("请输入问题后再发送")
        if len(current) > self.max_question_length:
            raise ValidationError("提问内容过长，请精简后重试")
        started = time.perf_counter()
        retrieval_query = self.query_processor.process(current, self.memory.user_queries())
        retrieval_started = time.perf_counter()
        candidates = self.retriever.search(retrieval_query)
        retrieval_seconds = time.perf_counter() - retrieval_started
        if not candidates:
            return AnswerResult(
                answer=NO_RESULT_MESSAGE,
                status="no_result",
                timings={"retrieval": retrieval_seconds, "total": time.perf_counter() - started},
            )
        rerank_started = time.perf_counter()
        references = self.reranker.rerank(retrieval_query, candidates)
        rerank_seconds = time.perf_counter() - rerank_started
        if not references:
            return AnswerResult(
                answer=NO_RESULT_MESSAGE,
                status="no_result",
                timings={
                    "retrieval": retrieval_seconds,
                    "rerank": rerank_seconds,
                    "total": time.perf_counter() - started,
                },
            )
        generation_started = time.perf_counter()
        answer = self.llm.generate(
            current_query=current,
            context=build_context(references),
            history=self.memory.messages(),
        )
        generation_seconds = time.perf_counter() - generation_started
        self.memory.add_turn(current, answer)
        return AnswerResult(
            answer=answer,
            references=references,
            timings={
                "retrieval": retrieval_seconds,
                "rerank": rerank_seconds,
                "generation": generation_seconds,
                "total": time.perf_counter() - started,
            },
        )
