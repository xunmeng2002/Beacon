# Beacon 简历 RAG 示例

端到端简历问答流水线：**简历文档 → 切片 → 本地 embedding → Beacon 入库 → 检索 Top-K → LLM 回答**。首次在真实循环中验证 pybind11 绑定。

## 依赖

```bash
pip install sentence-transformers     # torch 已装，仅补包
```

- embedding：`BAAI/bge-large-zh-v1.5`（dim=1024，余弦语义）。模型解析顺序：显式参数 > `BGE_MODEL_DIR` 环境变量 > 本地 `models/bge-large-zh-v1.5/`（有则离线加载）> 联网下载。
- **国内网络预下载模型（推荐）**：新版 huggingface_hub 与 hf-mirror 的重定向校验不兼容（`Distant resource does not seem to be on huggingface.co`），别用 `HF_ENDPOINT`，改用 curl 直接拉取到本地：

```bash
mkdir -p examples/rag/models/bge-large-zh-v1.5/1_Pooling
cd examples/rag/models/bge-large-zh-v1.5
base=https://hf-mirror.com/BAAI/bge-large-zh-v1.5/resolve/main
for f in config.json config_sentence_transformers.json modules.json sentence_bert_config.json \
         special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt 1_Pooling/config.json; do
  curl -L -o "$f" "$base/$f"
done
curl -L -o pytorch_model.bin "$base/pytorch_model.bin"     # ~1.2GB，耐心等待
```

`models/` 已 gitignore，不会提交。

- LLM：OpenAI 兼容接口，走标准库 `urllib`（零额外依赖），DeepSeek / 通义均可，改环境变量切换：

| 环境变量 | 说明 | 示例 |
| :--- | :--- | :--- |
| `RAG_LLM_BASE_URL` | 端点 | `https://api.deepseek.com/v1` 或通义 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `RAG_LLM_API_KEY` | API 密钥 | 缺失时 `ask` 自动降级为纯检索展示 |
| `RAG_LLM_MODEL` | 模型名 | `deepseek-chat` / `qwen-plus` |

## 用法

前置：先构建出 Python 绑定产物 `bin/Release/beacon.cp311-win_amd64.pyd`（见仓库根 README「Python 绑定」）。

```bash
# 1. 导入简历建库（docs/ 为样例，可替换成真实简历 .txt）
python examples/rag/demo.py import examples/rag/docs --out resume.beacon

# 2. 问答（未配置 API key 时仅展示检索 Top-K 命中）
python examples/rag/demo.py ask "该候选人的技术栈是什么" --db resume.beacon -k 5

# 3. 无子命令：建库后进入交互式问答
python examples/rag/demo.py import examples/rag/docs
```

## 目录与模块

| 文件 | 职责 |
| :--- | :--- |
| `chunking.py` | 纯文本切片：段落 + 句边界，`max_chars` / `overlap` 参数，确定性输出 |
| `embedder.py` | bge 中文 embedding 封装：passage 直接编码、query 带官方检索指令前缀 |
| `vector_store.py` | Beacon 门面封装：`add_text` / `search` / `save` / `load`，embed 函数注入便于测试打桩 |
| `llm.py` | OpenAI 兼容 chat（stdlib urllib）+ RAG 提示词拼接 |
| `demo.py` | CLI 入口（`import` / `ask` 子命令） |
| `test_rag.py` | 离线 pytest：chunking 确定性 + 打桩 embedder 的 store + LLM 提示词（不打模型不联网） |

## 测试

```bash
python -m pytest examples/rag/test_rag.py -v
```

## 已知局限（示例定位）

- 每次运行重新加载 embedding 模型（~5-10s），未做模型缓存。
- 切片、检索参数为默认值（`max_chars=256`、`overlap=32`、`k=5`），可按需调整。
- 简历需为 `.txt`（UTF-8）；PDF/DOCX 解析未接入。
- 展示用，非生产：并发、持久化原子性、混合检索（BM25）等见仓库路线图。
