import unittest

from rag_app.exceptions import ValidationError
from rag_app.parsing.web_parser import WebParser


class _Response:
    status_code = 200
    headers = {"Content-Type": "text/html; charset=utf-8", "Content-Length": "200"}
    content = """
    <html><body><header>站点导航</header><main><h1>休假制度</h1>
    <p>员工每年享有带薪年假。</p></main><script>secret()</script></body></html>
    """.encode()

    def raise_for_status(self):
        return None


class _Session:
    def get(self, *args, **kwargs):
        return _Response()


class WebParserTests(unittest.TestCase):
    def test_extracts_main_text_and_removes_navigation(self):
        parsed = WebParser(session=_Session()).parse("https://example.com/policy")
        self.assertIn("员工每年享有带薪年假", parsed.text)
        self.assertNotIn("站点导航", parsed.text)
        self.assertNotIn("secret", parsed.text)

    def test_rejects_non_http_url(self):
        with self.assertRaises(ValidationError):
            WebParser(session=_Session()).parse("file:///etc/passwd")


if __name__ == "__main__":
    unittest.main()
