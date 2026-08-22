import unittest

from rag_app.chunking.classifier import ChunkProfile, classify_document


class ClassifierTests(unittest.TestCase):
    def test_markdown_code_fence_is_technical(self):
        text = "接口文档\n```python\ndef answer(query):\n    return query\n```"
        self.assertEqual(classify_document(text, "md"), ChunkProfile.TECHNICAL)

    def test_business_prose_is_normal(self):
        text = "本制度适用于公司全体员工。员工提交申请后，由部门负责人进行审批。"
        self.assertEqual(classify_document(text, "docx"), ChunkProfile.NORMAL)


if __name__ == "__main__":
    unittest.main()
