"""bge 中文 embedding 封装：passage 直接编码、query 带官方检索指令前缀（提升召回）。

模型解析顺序：显式 model_name > 环境变量 BGE_MODEL_DIR > 本目录 models/ 下的本地副本 > 联网下载。
本目录 models/bge-large-zh-v1.5/ 可手动预下载（国内网络友好，见 README），有则离线加载。
依赖 sentence-transformers（torch 已装）；本模块独立，测试不打桩时不 import 它。
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

# bge 官方检索指令：query 侧必须加此前缀，passage 侧不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DEFAULT_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
_LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "bge-large-zh-v1.5"))


def _resolve_model(model_name: str | None) -> str:
    if model_name:
        return model_name
    env_dir = os.environ.get("BGE_MODEL_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    if os.path.isdir(_LOCAL_MODEL_DIR):
        return _LOCAL_MODEL_DIR
    return DEFAULT_MODEL_NAME


class BgeEmbedder:
    """BAAI bge 系列中文 embedding（余弦相似度语义，编码时归一化）。"""

    def __init__(self, model_name: str | None = None) -> None:
        self._model = SentenceTransformer(_resolve_model(model_name))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """文档块编码：不加指令前缀，返回 (n, dim) 归一化向量。"""
        return self._model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """查询编码：加检索指令前缀，返回 (1, dim) 归一化向量。"""
        return self._model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)

    @property
    def dim(self) -> int:
        # ST 6.0 起更名，兼容旧版用 getattr 回退
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        return get_dim()
