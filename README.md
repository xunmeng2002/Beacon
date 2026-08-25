# MdbVector

自研的**内存向量数据库**（C++17）。为 RAG（检索增强生成）场景提供基于余弦相似度 / 内积的 Top-K 检索，并支持二进制持久化。作为个人项目，重点在于吃透向量索引与检索的底层原理，而非堆砌依赖。

## 特性

- **扁平连续存储**：向量按 `count × dim` 连续存放，缓存友好，`O(1)` 按 id 访问
- **两种度量**：余弦相似度（文本 embedding 默认）、内积
  - 余弦模式在**插入时** L2 归一化，检索时只需一次点积，避免每次查询重复算范数
- **SIMD 加速**：`x86_64` 上点积走 **AVX2**（8 路 float 并行），其余平台自动标量回退
- **暴力 Top-K**：小规模数据直接全扫描 + 最小堆维护 K 个最优，结果按分数降序返回
- **二进制持久化**：`magic("MDBV") + version + dim + metric + count + 数据 + 元数据`，加载时校验魔数与版本
- **元数据**：每个向量可携带任意字符串（文档 id、原文片段等）

## 构建

需要 **CMake ≥ 3.16** 和任一 C++17 编译器（MSVC / GCC / Clang）。

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64   # Windows / MSVC
# 或
cmake -S . -B build                                     # 默认生成器
cmake --build build --config Release
```

> x86_64 平台自动追加 `/arch:AVX2`（MSVC）或 `-mavx2`（GCC/Clang）。

## 运行 Demo

```bash
./build/Release/mdbvec_demo.exe        # Windows
./build/mdbvec_demo                    # Linux/macOS
```

Demo 输出两类结果：

1. **3 维余弦检索 + 持久化往返**：验证归一化、Top-K 排序、save/load 正确性
2. **10k × 64 维暴力检索延迟**：量测全扫描耗时（AVX2 下约 0.2~0.4 ms）

## 目录结构

```
include/mdbvec/
  Metrics.h       距离度量：点积、L2 范数、L2 归一化
  VectorTable.h   定长向量表：扁平存储 + 插入时归一化
  VectorDb.h      门面：搜索、持久化、清空
src/
  Metrics.cpp     AVX2 / 标量双路径点积
  VectorTable.cpp
  VectorDb.cpp    最小堆 Top-K、Save/Load
  main.cpp        demo
```

## 快速上手

```cpp
#include "mdbvec/vector_db.h"

mdbvec::VectorDb db(384, mdbvec::Metric::kCosine);
db.Add({ 0.1f, 0.2f, /* ... */ }, "文档A");

auto hits = db.Search({ 0.15f, 0.21f, /* ... */ }, 5);  // 返回 Top-5
db.Save("index.mdbv");                                  // 持久化
db.Load("index.mdbv");                                  // 恢复
```

## 路线图

- [ ] **HNSW 索引**：跳过图结构，把检索从 `O(n·dim)` 降到亚线性，支持百万级规模
- [ ] **pybind11 绑定**：提供 Python API，供 RAG 流水线直接调用
- [ ] **混合检索**：结合 BM25 关键词检索，文本召回更稳
- [ ] **RAG 示例**：金融文档切片 → embedding → 本库检索 → 拼接到 LLM 提示词
- [ ] **mmap 持久化**：大文件免加载全部进内存，支持冷启动即查
- [ ] **基准数据**：与 faiss / hnswlib 对比召回率与延迟，产出简历可用数据
