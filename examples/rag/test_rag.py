"""RAG 示例离线 pytest：chunking 确定性用例 + vector_store 用打桩 embedder + LLM 提示词拼接。

不打模型、不联网。需先构建出 bin/Release/beacon*.pyd。
运行：python -m pytest examples/rag/test_rag.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from chunking import chunk_text, split_paragraphs, split_sentences
from llm import LlmConfig, build_rag_messages
from vector_store import VectorStore

STUB_DIM = 4


def _stub_embedder(dim: int = STUB_DIM) -> tuple:
    """确定性打桩：文本首字符哈希到 dim 维 one-hot 向量，query 与 passage 同映射。"""

    def _vector_for(text: str) -> list[float]:
        vec = [0.0] * dim
        vec[min(ord(text[0]) % dim, dim - 1)] = 1.0
        return vec

    def embed_passages(texts: list[str]) -> np.ndarray:
        return np.array([_vector_for(t) for t in texts], dtype=np.float32)

    def embed_query(query: str) -> np.ndarray:
        return np.array([_vector_for(query)], dtype=np.float32)

    return embed_passages, embed_query


class TestChunking:
    def test_split_paragraphs_splits_on_blank_lines(self) -> None:
        text = "第一段。\n\n第二段。\n第三段。"
        assert split_paragraphs(text) == ["第一段。", "第二段。\n第三段。"]

    def test_split_sentences_keeps_punctuation(self) -> None:
        assert split_sentences("你好。再见！") == ["你好。", "再见！"]

    def test_chunk_empty_text_returns_empty_list(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("  \n  ") == []

    def test_chunk_short_text_keeps_single_chunk(self) -> None:
        text = "张三，后端工程师。精通 C++。"
        assert chunk_text(text) == [text]

    def test_chunk_long_paragraph_splits_within_max_chars(self) -> None:
        text = "项目经历。" + "我们采用 C++20 实现内存向量数据库，支持 HNSW 索引与精确检索。" * 6
        chunks = chunk_text(text, max_chars=64, overlap=0)
        assert len(chunks) >= 2
        assert all(len(chunk) <= 64 for chunk in chunks)

    def test_chunk_overlap_links_adjacent_chunks(self) -> None:
        long_text = "我们使用 C++ 与 CMake 构建，并接入 GoogleTest 单元测试。" * 8
        text = "介绍技术栈。" + long_text
        chunks = chunk_text(text, max_chars=80, overlap=16)
        assert len(chunks) >= 2
        assert chunks[1].startswith(chunks[0][-16:])

    def test_chunk_preserves_line_and_paragraph_structure(self) -> None:
        text = "基本信息\n联系方式：13800000001\n\n专业技能\n- 精通 C++20\n- 熟悉 HNSW"
        assert chunk_text(text, overlap=0) == [
            "基本信息\n联系方式：13800000001\n\n专业技能\n- 精通 C++20\n- 熟悉 HNSW"
        ]


class TestVectorStoreStub:
    def test_search_returns_nearest_chunk_with_parsed_metadata(self) -> None:
        embed_passages, embed_query = _stub_embedder()
        store = VectorStore(dim=STUB_DIM, embed_passages=embed_passages, embed_query=embed_query)
        store.add_text("甲：精通 C++ 与 CMake，五年后端经验。", source="resume_a.txt")
        store.add_text("乙：熟悉 Python 与机器学习，两年算法经验。", source="resume_b.txt")

        hits = store.search("甲：后端工程师", k=1)
        assert len(hits) == 1
        assert hits[0]["source"] == "resume_a.txt#0"  # 来源精确到块（文件名#块号）
        assert "C++" in hits[0]["text"]

    def test_search_no_index_falls_back_to_exact(self) -> None:
        embed_passages, embed_query = _stub_embedder()
        store = VectorStore(dim=STUB_DIM, embed_passages=embed_passages, embed_query=embed_query)
        store.add_text("甲：精通 C++。", source="resume_a.txt")
        hits = store.search("甲：精通 C++。", k=5)
        assert hits[0]["text"] == "甲：精通 C++。"

    def test_add_returns_chunk_count(self) -> None:
        embed_passages, embed_query = _stub_embedder()
        store = VectorStore(dim=STUB_DIM, embed_passages=embed_passages, embed_query=embed_query)
        assert store.add_text("甲：一句话。", source="a.txt") == 1
        assert store.add_text("", source="empty.txt") == 0


class TestLlm:
    def test_build_rag_messages_embeds_hits_and_question(self) -> None:
        hits = [
            {"id": 0, "score": 0.9, "source": "a.txt", "text": "精通 C++"},
            {"id": 1, "score": 0.8, "source": "b.txt", "text": "熟悉 Python"},
        ]
        messages = build_rag_messages("技术栈是什么", hits, k=2)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "技术栈是什么" in messages[1]["content"]
        assert "精通 C++" in messages[1]["content"]
        assert "b.txt" in messages[1]["content"]

    def test_build_rag_messages_honors_k(self) -> None:
        hits = [
            {"id": i, "score": 0.9, "source": f"{i}.txt", "text": f"内容{i}"}
            for i in range(5)
        ]
        messages = build_rag_messages("问题", hits, k=2)
        assert "内容0" in messages[1]["content"]
        assert "内容1" in messages[1]["content"]
        assert "内容2" not in messages[1]["content"]

    def test_llm_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_LLM_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("RAG_LLM_API_KEY", "secret")
        monkeypatch.setenv("RAG_LLM_MODEL", "qwen-plus")
        cfg = LlmConfig.from_env()
        assert cfg.base_url == "https://example.com/v1"
        assert cfg.api_key == "secret"
        assert cfg.model == "qwen-plus"

    def test_llm_config_defaults_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("RAG_LLM_BASE_URL", "RAG_LLM_API_KEY", "RAG_LLM_MODEL"):
            monkeypatch.delenv(name, raising=False)
        cfg = LlmConfig.from_env()
        assert cfg.api_key is None
        assert cfg.base_url.endswith("api.deepseek.com/v1")
