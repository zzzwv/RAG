from __future__ import annotations

import streamlit as st

from rag_app.chat.memory import WindowMemory
from rag_app.config import AppSettings
from rag_app.exceptions import RAGError
from rag_app.logging_config import read_recent_logs
from rag_app.runtime import build_runtime
from rag_app.security.admin_auth import verify_admin_password
from rag_app.ui.helpers import format_reference, profile_from_label


@st.cache_resource(show_spinner="正在加载知识库与模型配置…")
def get_runtime():
    return build_runtime()


def _init_state(settings: AppSettings) -> None:
    if "memory" not in st.session_state or st.session_state.get("memory_turns") != settings.memory_turns:
        st.session_state.memory = WindowMemory(settings.memory_turns, settings.memory_max_chars)
        st.session_state.memory_turns = settings.memory_turns
        st.session_state.chat_history = []
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("admin_unlocked", False)
    st.session_state.setdefault("busy", False)


def _show_references(references) -> None:
    if not references:
        return
    with st.expander("查看知识库引用", expanded=False):
        for item in references:
            st.caption(format_reference(item))
            st.markdown(item.chunk.content)
            st.divider()


def _render_ingestion(runtime) -> None:
    st.subheader("知识库")
    profile_label = st.selectbox("切片策略", ["自动判断", "普通文档", "技术文档"])
    profile = profile_from_label(profile_label)
    files = st.file_uploader(
        "上传 PDF / Word / MD / TXT",
        type=["pdf", "doc", "docx", "md", "txt"],
        accept_multiple_files=True,
        disabled=st.session_state.busy,
    )
    if st.button("导入文件", disabled=st.session_state.busy or not files, use_container_width=True):
        st.session_state.busy = True
        progress = st.progress(0, text="准备解析")
        successes = 0
        for index, uploaded in enumerate(files):
            try:
                result = runtime.ingestion.ingest_file(uploaded.name, uploaded.getvalue(), profile)
                successes += 1
                runtime.logger.info("导入成功 source=%s chunks=%s", result.source, result.chunk_count)
                st.success(f"{result.source}：已导入 {result.chunk_count} 个切片")
            except Exception as exc:
                runtime.logger.exception("文件导入失败 source=%s", uploaded.name)
                message = str(exc) if isinstance(exc, RAGError) else "文档解析失败，请检查文件格式与完整性"
                st.error(f"{uploaded.name}：{message}")
            progress.progress((index + 1) / len(files), text=f"已处理 {index + 1}/{len(files)}")
        st.session_state.busy = False
        if successes:
            st.toast(f"成功导入 {successes} 个文件")

    url = st.text_input("网页链接", placeholder="https://intranet.example.com/guide")
    if st.button("解析网页", disabled=st.session_state.busy or not url.strip(), use_container_width=True):
        st.session_state.busy = True
        try:
            with st.spinner("正在抓取并构建索引…"):
                result = runtime.ingestion.ingest_url(url.strip(), profile)
            runtime.logger.info("网页导入成功 source=%s chunks=%s", result.source, result.chunk_count)
            st.success(f"网页已导入，共 {result.chunk_count} 个切片")
        except Exception as exc:
            runtime.logger.exception("网页导入失败")
            message = str(exc) if isinstance(exc, RAGError) else "网页内容获取失败，请更换链接重试"
            st.error(message)
        finally:
            st.session_state.busy = False

    st.metric("知识库切片", runtime.vector_store.count())
    st.caption(
        f"向量 + BM25 / RRF k={runtime.settings.rrf_k} / "
        f"重排 {'开启' if runtime.reranker else '关闭'}"
    )


def _render_admin(runtime) -> None:
    with st.expander("运维管理", expanded=False):
        if not runtime.settings.admin_password_hash:
            st.warning("尚未设置运维口令，请运行 scripts/manage_secrets.py init-admin")
        elif not st.session_state.admin_unlocked:
            password = st.text_input("运维口令", type="password")
            if st.button("解锁运维功能"):
                if verify_admin_password(password, runtime.settings.admin_password_hash):
                    st.session_state.admin_unlocked = True
                    st.rerun()
                st.error("运维口令错误")
        else:
            st.success("运维功能已解锁（仅当前会话）")
            with st.form("advanced_settings"):
                normal_size = st.number_input("普通文档 chunk_size", 100, 5000, runtime.settings.normal_chunk_size)
                normal_overlap = st.number_input("普通文档 overlap", 0, 1000, runtime.settings.normal_chunk_overlap)
                technical_size = st.number_input("技术文档 chunk_size", 100, 5000, runtime.settings.technical_chunk_size)
                technical_overlap = st.number_input("技术文档 overlap", 0, 1000, runtime.settings.technical_chunk_overlap)
                memory_turns = st.number_input("记忆轮数", 1, 50, runtime.settings.memory_turns)
                threshold = st.slider("重排阈值", 0.0, 1.0, runtime.settings.rerank_threshold, 0.01)
                llm_base_url = st.text_input("LLM base_url", runtime.settings.llm_base_url)
                llm_model = st.text_input("LLM model", runtime.settings.llm_model)
                if st.form_submit_button("保存并重新加载"):
                    runtime.settings.normal_chunk_size = int(normal_size)
                    runtime.settings.normal_chunk_overlap = int(normal_overlap)
                    runtime.settings.technical_chunk_size = int(technical_size)
                    runtime.settings.technical_chunk_overlap = int(technical_overlap)
                    runtime.settings.memory_turns = int(memory_turns)
                    runtime.settings.rerank_threshold = threshold
                    runtime.settings.llm_base_url = llm_base_url.strip()
                    runtime.settings.llm_model = llm_model.strip()
                    try:
                        runtime.settings.validate().save()
                        get_runtime.clear()
                        st.success("配置已保存")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            confirm = st.checkbox("我确认清空全部知识库数据")
            if st.button("清空知识库", type="primary", disabled=not confirm):
                runtime.ingestion.clear()
                runtime.logger.warning("知识库已由运维用户清空")
                st.success("知识库已清空")
                st.rerun()


def _render_chat(runtime) -> None:
    title_col, clear_col = st.columns([4, 1])
    title_col.subheader("企业知识问答")
    if clear_col.button("清空对话", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            _show_references(turn["references"])
            with st.expander("复制回答纯文本"):
                st.code(turn["answer"], language=None, wrap_lines=True)

    question = st.chat_input("请输入问题（最多 500 字）", disabled=st.session_state.busy)
    if question is not None:
        st.session_state.busy = True
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"):
                with st.spinner("正在检索、重排并生成回答…"):
                    result = runtime.create_qa(st.session_state.memory).answer(question)
                st.markdown(result.answer)
                _show_references(result.references)
                st.caption(" · ".join(f"{key}: {value:.2f}s" for key, value in result.timings.items()))
            st.session_state.chat_history.append(
                {"question": question.strip(), "answer": result.answer, "references": result.references}
            )
            runtime.logger.info("问答完成 status=%s total=%.3f", result.status, result.timings.get("total", 0))
        except Exception as exc:
            runtime.logger.exception("问答失败")
            st.error(str(exc) if isinstance(exc, RAGError) else "问答服务异常，请稍后重试")
        finally:
            st.session_state.busy = False


def render_page() -> None:
    st.set_page_config(page_title="企业知识库智能问答", page_icon="📚", layout="wide")
    st.title("企业知识库智能问答系统")
    st.caption("混合检索 · RRF 融合 · BGE 重排 · 窗口对话记忆")
    try:
        runtime = get_runtime()
    except Exception:
        st.error("系统初始化失败，请运维人员检查依赖、模型和配置。")
        st.exception(Exception("可运行 scripts/check_environment.py 获取诊断信息"))
        return
    _init_state(runtime.settings)
    left, right = st.columns([1, 2], gap="large")
    with left:
        _render_ingestion(runtime)
        _render_admin(runtime)
        with st.expander("最近运行日志"):
            st.code("\n".join(read_recent_logs(runtime.settings.log_directory)) or "暂无日志", language="text")
    with right:
        _render_chat(runtime)
