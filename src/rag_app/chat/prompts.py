SYSTEM_PROMPT = """你是企业内部知识库问答助手。只能依据提供的参考资料回答。
若资料不足，明确说明知识库中没有足够信息；禁止补充未经资料支持的事实。
回答使用简洁中文，并在相关内容后标注 [来源N]。"""


def build_context(references) -> str:
    blocks = []
    for index, hit in enumerate(references, start=1):
        blocks.append(
            f"[来源{index}] 文件: {hit.chunk.source}; 位置: {hit.chunk.location}\n{hit.chunk.content}"
        )
    return "\n\n".join(blocks)
