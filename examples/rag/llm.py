"""OpenAI 兼容 chat（stdlib urllib，零第三方依赖）：DeepSeek / 通义等改环境变量即可切换。

环境变量：
    RAG_LLM_BASE_URL   端点，如 https://api.deepseek.com/v1 或
                       通义 https://dashscope.aliyuncs.com/compatible-mode/v1
    RAG_LLM_API_KEY    API 密钥（缺失时 demo 降级为纯检索展示）
    RAG_LLM_MODEL      模型名，如 deepseek-chat / qwen-plus
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 60


@dataclass
class LlmConfig:
    base_url: str
    api_key: str | None
    model: str

    @classmethod
    def from_env(cls) -> "LlmConfig":
        return cls(
            base_url=os.environ.get("RAG_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=os.environ.get("RAG_LLM_API_KEY"),
            model=os.environ.get("RAG_LLM_MODEL", DEFAULT_MODEL),
        )


def chat(
    cfg: LlmConfig,
    messages: list[dict[str, str]],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """调用 OpenAI 兼容 /chat/completions，返回助手回复文本。"""
    if not cfg.api_key:
        raise ValueError("缺少 RAG_LLM_API_KEY，无法调用 LLM")
    payload: dict = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{cfg.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM 请求失败 HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM 网络错误：{exc.reason}") from exc
    choices = body.get("choices")
    if not choices:
        raise RuntimeError(f"LLM 响应缺少 choices：{body}")
    return choices[0]["message"]["content"]


def build_rag_messages(question: str, hits: list[dict], k: int) -> list[dict[str, str]]:
    """把 Top-K 命中拼进提示词：system 限定依据范围，user 携带带来源的上下文与问题。"""
    context_lines = [
        f"[{rank}] 来源 {hit['source']}\n{hit['text']}"
        for rank, hit in enumerate(hits[:k], start=1)
    ]
    context = "\n\n".join(context_lines)
    system = "你是简历信息助手，只依据下方提供的简历内容回答问题；内容中没有的信息，请明确说明未提及。"
    user = f"以下是与问题相关的简历片段：\n\n{context}\n\n问题：{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
