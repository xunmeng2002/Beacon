"""Beacon 门面封装：文本 → 切片 → embedding → 入库 → 检索。

embed 函数以参数注入，测试可传 stub（不下载模型）；metadata 存 "{source}#{idx}\\n{原文}"，
检索后直接解析出来源与原文，且随 save/load 持久化。
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np

# 让 `import beacon` 命中构建产物 bin/Release/beacon*.pyd（仓库根 = examples/rag 的上两级）
_BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "Release"))
if _BIN_DIR not in sys.path:
    sys.path.insert(0, _BIN_DIR)

import beacon  # noqa: E402

from chunking import chunk_text  # noqa: E402

EmbedFunc = Callable[[list[str]], np.ndarray]


class VectorStore:
    def __init__(self, dim: int, embed_passages: EmbedFunc, embed_query: EmbedFunc) -> None:
        self._dim = dim
        self._embed_passages = embed_passages
        self._embed_query = embed_query
        self._db = beacon.VectorDb(dim, beacon.Metric.kCosine)

    def add_text(self, text: str, source: str) -> int:
        """切块 → 批量 embedding → 逐块入库，返回入库块数。"""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        vectors = self._embed_passages(chunks)
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self._db.add(vector.tolist(), f"{source}#{idx}\n{chunk}")
        return len(chunks)

    def search(self, query: str, k: int = 5, ef: int = 100) -> list[dict]:
        """检索 Top-K，返回 [{id, score, source, text}]；有索引走近似、否则精确。"""
        query_vector = self._embed_query(query).flatten()
        if self._db.index_enabled():
            hits = self._db.search_indexed(query_vector.tolist(), k, ef)
        else:
            hits = self._db.search_exact(query_vector.tolist(), k)
        results: list[dict] = []
        for hit in hits:
            source, _, text = self._db.metadata(hit.id).partition("\n")
            results.append({"id": hit.id, "score": hit.score, "source": source, "text": text})
        return results

    def enable_index(self) -> None:
        self._db.enable_index()

    def save(self, path: str) -> bool:
        return self._db.save(path)

    @classmethod
    def load(cls, path: str, dim: int, embed_passages: EmbedFunc, embed_query: EmbedFunc) -> VectorStore:
        store = cls(dim, embed_passages, embed_query)
        if not store._db.load(path):
            raise ValueError(f"无法加载数据库文件：{path}")
        return store
