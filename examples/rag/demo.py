"""Beacon 简历 RAG 演示 CLI：`import` 建库 / `ask` 问答（无 API key 自动降级为纯检索展示）。

用法：
    python examples/rag/demo.py import examples/rag/docs --out resume.beacon
    python examples/rag/demo.py ask "该候选人的技术栈是什么" --db resume.beacon -k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from embedder import BgeEmbedder
from llm import LlmConfig, build_rag_messages, chat
from vector_store import VectorStore


def _load_embedder() -> BgeEmbedder:
    print("加载 bge 中文 embedding 模型（优先用本地 models/ 目录，无则联网下载）...")
    return BgeEmbedder()


def _import_docs(docs_dir: str, out_path: str) -> None:
    embedder = _load_embedder()
    store = VectorStore(embedder.dim, embedder.embed_passages, embedder.embed_query)
    doc_dir = Path(docs_dir)
    total_chunks = 0
    for doc_path in sorted(path for path in doc_dir.glob("*.txt") if path.is_file()):
        text = doc_path.read_text(encoding="utf-8")
        added = store.add_text(text, source=doc_path.name)
        total_chunks += added
        print(f"  {doc_path.name}: {added} 块")
    if total_chunks == 0:
        print(f"未在 {doc_dir} 找到任何 .txt 简历，未建库。")
        return
    store.enable_index()
    if not store.save(out_path):
        raise RuntimeError(f"保存失败：{out_path}")
    print(f"建库完成：共 {total_chunks} 块，dim={embedder.dim}，索引已启用，已保存 {out_path}")


def _ask(db_path: str, question: str, k: int, ef: int) -> None:
    embedder = _load_embedder()
    store = VectorStore.load(db_path, embedder.dim, embedder.embed_passages, embedder.embed_query)
    print(f"检索 Top-{k}：{question}\n")
    hits = store.search(question, k=k, ef=ef)
    for rank, hit in enumerate(hits, start=1):
        print(f"[{rank}] {hit['source']}  score={hit['score']:.4f}\n    {hit['text']}\n")
    if not hits:
        print("未检索到相关简历片段。")
        return
    cfg = LlmConfig.from_env()
    if cfg.api_key is None:
        print("未配置 RAG_LLM_API_KEY，仅展示检索结果（配置后可输出 LLM 回答）。")
        return
    print("调用 LLM 生成回答...\n")
    messages = build_rag_messages(question, hits, k)
    print(f"回答：{chat(cfg, messages)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beacon-rag", description="Beacon 简历 RAG 演示")
    sub = parser.add_subparsers(dest="command")

    cmd_import = sub.add_parser("import", help="导入简历文档建库")
    cmd_import.add_argument("docs_dir", help="简历目录（*.txt）")
    cmd_import.add_argument("--out", default="resume.beacon", help="数据库文件（默认 resume.beacon）")

    cmd_ask = sub.add_parser("ask", help="检索 + LLM 问答")
    cmd_ask.add_argument("question", help="问题")
    cmd_ask.add_argument("--db", default="resume.beacon", help="数据库文件")
    cmd_ask.add_argument("-k", "--top-k", type=int, default=5, help="返回 Top-K 命中")
    cmd_ask.add_argument("--ef", type=int, default=100, help="HNSW 检索 ef 参数")

    args = parser.parse_args(argv)
    if args.command == "import":
        _import_docs(args.docs_dir, args.out)
    elif args.command == "ask":
        _ask(args.db, args.question, args.top_k, args.ef)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
