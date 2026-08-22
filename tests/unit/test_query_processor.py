import unittest
import time

from rag_app.retrieval.query_processor import RetrievalQueryProcessor


class RetrievalQueryProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = RetrievalQueryProcessor(max_length=512)

    def test_new_topic_is_not_joined_with_history(self):
        result = self.processor.process("公司的报销制度是什么？", ["年假有几天？"])
        self.assertEqual(result, "公司的报销制度是什么？")

    def test_pronoun_query_joins_only_latest_user_question(self):
        result = self.processor.process("它可以延期吗？", ["旧问题", "年假有效期多长？"])
        self.assertEqual(result, "年假有效期多长？ 它可以延期吗？")

    def test_ellipsis_query_joins_latest_question(self):
        result = self.processor.process("怎么申请？", ["差旅报销有哪些条件？"])
        self.assertEqual(result, "差旅报销有哪些条件？ 怎么申请？")

    def test_missing_history_falls_back_to_current_query(self):
        self.assertEqual(self.processor.process("那个有效吗？", []), "那个有效吗？")

    def test_current_query_is_preserved_when_join_is_truncated(self):
        current = "它" + "x" * 498
        result = self.processor.process(current, ["主题" * 100])
        self.assertLessEqual(len(result), 512)
        self.assertTrue(result.endswith(current))

    def test_processing_average_is_under_five_milliseconds(self):
        started = time.perf_counter()
        for _ in range(10_000):
            self.processor.process("它可以延期吗？", ["年假有效期多长？"])
        average_ms = (time.perf_counter() - started) * 1000 / 10_000
        self.assertLess(average_ms, 5.0)


if __name__ == "__main__":
    unittest.main()
