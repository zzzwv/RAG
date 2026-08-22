from __future__ import annotations

from urllib.parse import urlparse

from rag_app.exceptions import InvalidContentError, ValidationError, WebParseError
from rag_app.models import DocumentUnit, ParsedDocument
from rag_app.parsing.cleaners import clean_text


class WebParser:
    def __init__(self, session=None, *, timeout: tuple[float, float] = (5.0, 15.0), max_bytes: int = 5 * 1024 * 1024) -> None:
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.timeout = timeout
        self.max_bytes = max_bytes

    def parse(self, url: str) -> ParsedDocument:
        parsed_url = urlparse(url.strip())
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValidationError("请输入有效的 HTTP/HTTPS 网页链接")
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "EnterpriseRAG/1.0"},
                allow_redirects=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type:
                raise WebParseError("网页内容获取失败，请更换链接重试")
            length = int(response.headers.get("Content-Length", len(response.content)))
            if length > self.max_bytes or len(response.content) > self.max_bytes:
                raise ValidationError("网页内容过大，请更换链接重试")
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup.select("script,style,noscript,nav,header,footer,aside,form,iframe,svg"):
                tag.decompose()
            root = soup.find("main") or soup.find("article") or soup.body or soup
            text = root.get_text("\n", strip=True)
            cleaned = clean_text(text)
        except (ValidationError, WebParseError, InvalidContentError):
            raise
        except Exception as exc:
            raise WebParseError("网页内容获取失败，请更换链接重试") from exc
        return ParsedDocument(
            source=url,
            doc_type="web",
            text=cleaned,
            units=[DocumentUnit(text=cleaned, location=parsed_url.path or "/")],
            metadata={"url": url},
        )
