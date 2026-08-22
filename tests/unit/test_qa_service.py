import unittest

from rag_app.chat.memory import WindowMemory
from rag_app.chat.service import NO_RESULT_MESSAGE, QAService
from rag_app.exceptions import LLMServiceError, ValidationError
from rag_app.models import ChunkRecord, SearchHit
from rag_app.retrieval.query_processor import RetrievalQueryProcessor


def hit() -> SearchHit:
    return SearchHit(chunk=ChunkRecord.create(
        source="policy.txt", doc_type="txt", content="年假为五天。", index=0,
        location="body", chunk_profile="normal",
    ), rerank_score=0.9)


class _Retriever:
    def __init__(self, results):
        self.results = results

    def search(self, query):
        return self.results


class _Reranker:
    def __init__(self, results):
        self.results = results

    def rerank(self, query, hits):
        return self.results


class _LLM:
    def __init__(self, answer="回答", error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


class QAServiceTests(unittest.TestCase):
    def build(self, retrieved, reranked, llm):
        return QAService(
            retriever=_Retriever(retrieved), reranker=_Reranker(reranked), llm=llm,
            memory=WindowMemory(10), query_processor=RetrievalQueryProcessor(),
        )

    def test_no_result_returns_fixed_message_without_calling_llm(self):
        llm = _LLM()
        service = self.build([], [], llm)
        result = service.answer("不存在的问题")
        self.assertEqual(result.answer, NO_RESULT_MESSAGE)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(service.memory.messages(), [])

    def test_success_stores_complete_turn_and_references(self):
        item = hit()
        llm = _LLM("年假为五天。")
        service = self.build([item], [item], llm)
        result = service.answer("年假有几天？")
        self.assertEqual(result.references, [item])
        self.assertEqual(service.memory.user_queries(), ["年假有几天？"])

    def test_llm_failure_does_not_store_partial_turn(self):
        item = hit()
        service = self.build([item], [item], _LLM(error=LLMServiceError("failed")))
        with self.assertRaises(LLMServiceError):
            service.answer("年假有几天？")
        self.assertEqual(service.memory.messages(), [])

    def test_empty_and_oversized_questions_are_rejected(self):
        service = self.build([], [], _LLM())
        with self.assertRaises(ValidationError):
            service.answer("   ")
        with self.assertRaises(ValidationError):
            service.answer("x" * 501)


if __name__ == "__main__":
    unittest.main()
