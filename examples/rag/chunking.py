"""简历文档切片：按段落 + 句边界切分，相邻块带重叠窗口（纯标准库，确定性输出，便于单测）。"""

from __future__ import annotations

import re

# 句边界含 \n：换行既是分句点也保留在句末，块内完整还原简历的按行排版
_SENTENCE_BOUNDARY_CHARS = "。！？…；\n"
_BLANK_LINE_PATTERN = re.compile(r"\n[ \t]*\n")


def split_paragraphs(text: str) -> list[str]:
    """按空行分段落，去掉空段；段落内保留原有换行。"""
    blocks = _BLANK_LINE_PATTERN.split(text)
    return [block.strip() for block in blocks if block.strip()]


def split_sentences(text: str) -> list[str]:
    """按句尾标点或换行分句；边界字符（含 \n）保留在句末，拼接后可还原原文换行结构。"""
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in _SENTENCE_BOUNDARY_CHARS:
            sentences.append("".join(current))
            current = []
    remainder = "".join(current)
    if remainder.strip():
        sentences.append(remainder)
    return sentences


def chunk_text(text: str, *, max_chars: int = 256, overlap: int = 32) -> list[str]:
    """按段落、句边界切块；相邻块重叠 overlap 字符衔接上下文，防止语义在边界处割裂。

    - 每块不超过 max_chars（超长句按 max_chars 硬切；overlap 装不下时放弃重叠）
    - 确定性输出，同一输入恒得同一结果
    """
    if not text.strip():
        return []
    chunks: list[str] = []
    for paragraph_index, paragraph in enumerate(split_paragraphs(text)):
        for sentence_index, sentence in enumerate(split_sentences(paragraph)):
            if not sentence.strip():
                continue
            if paragraph_index > 0 and sentence_index == 0:
                sentence = "\n\n" + sentence  # 段落间空行已被 split_paragraphs 吞掉，句首补回
            if len(sentence) > max_chars:
                for start in range(0, len(sentence), max_chars):
                    chunks.append(sentence[start : start + max_chars])
                continue
            if not chunks:
                chunks.append(sentence)
                continue
            tail = chunks[-1][-overlap:] if overlap > 0 else ""
            if len(tail) + len(sentence) > max_chars:
                tail = ""  # overlap 会顶爆块上限时放弃重叠，保块长不变量
            if len(chunks[-1]) + len(sentence) > max_chars:
                chunks.append(tail + sentence)
            else:
                chunks[-1] += sentence
    return chunks
