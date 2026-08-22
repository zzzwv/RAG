from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_ui_exception():
    app = AppTest.from_file("app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "企业知识库智能问答系统"
