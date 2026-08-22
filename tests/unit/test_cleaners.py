import unittest

from rag_app.exceptions import InvalidContentError
from rag_app.parsing.cleaners import clean_text, remove_repeated_page_lines


class CleanerTests(unittest.TestCase):
    def test_clean_text_normalizes_controls_whitespace_and_duplicates(self):
        raw = "标题\x00\n\n 正文  内容\n重复行\n重复行"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "标题\n正文 内容\n重复行")

    def test_clean_text_rejects_replacement_character_noise(self):
        with self.assertRaises(InvalidContentError):
            clean_text("有效" + "�" * 20)

    def test_repeated_header_and_footer_are_removed(self):
        pages = [
            "企业手册\n第一页正文\n第 1 页",
            "企业手册\n第二页正文\n第 2 页",
            "企业手册\n第三页正文\n第 3 页",
        ]
        result = remove_repeated_page_lines(pages)
        self.assertNotIn("企业手册", "\n".join(result))
        self.assertIn("第二页正文", result[1])


if __name__ == "__main__":
    unittest.main()
