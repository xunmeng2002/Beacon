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
- **CRUD**：软删除（tombstone，数据不动、id 稳定）+ 就地更新（对已删除 id 执行则复活）
- **HNSW 索引**：分层可导航小世界图，近似检索（`ef` 可调）；Delete/Update 走节点级增量维护（重连邻居保连通），暴力检索保留作精确对照
- **持久化格式版本化**：v2 起记录 tombstone；v3 起含 HNSW 索引段（向量段权威、索引段为可校验缓存，损坏即降级懒重建）；向后兼容 v1/v2

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

Demo 输出三类结果：

1. **3 维余弦检索 + 持久化往返**：验证归一化、Top-K 排序、软删除/更新、save/load 正确性
2. **10k × 64 维暴力检索延迟**：量测全扫描耗时（AVX2 下约 0.2 ms）
3. **30k × 64 维 HNSW vs 暴力**：recall@10 与延迟对比 + 增量删 3000 节点耗时（毫秒级）+ 含索引持久化重载免重建验证

## 目录结构

```
include/mdbvec/
  Metrics.h       距离度量：点积、L2 范数、L2 归一化
  VectorTable.h   定长向量表：扁平存储 + 归一化 + 软删除/就地更新
  HnswIndex.h     HNSW 分层小世界图索引（近似检索）
  VectorDb.h      门面：精确/近似搜索、持久化、清空
src/
  Metrics.cpp     AVX2 / 标量双路径点积
  VectorTable.cpp
  HnswIndex.cpp   HNSW 建图/检索
  VectorDb.cpp    最小堆 Top-K、Save/Load
  main.cpp        demo
```

## 快速上手

```cpp
#include "mdbvec/VectorDb.h"

mdbvec::VectorDb db(384, mdbvec::Metric::kCosine);
db.Add({ 0.1f, 0.2f, /* ... */ }, "文档A");

db.Update(id, { 0.2f, 0.1f, /* ... */ }, "文档A(修订)");  // 就地更新
db.Delete(id);                                            // 软删除，id 仍有效

auto hits = db.Search({ 0.15f, 0.21f, /* ... */ }, 5);  // 精确 Top-5

db.EnableIndex();                                       // 构建 HNSW 索引
auto near = db.SearchIndexed({ 0.15f, 0.21f, /* ... */ }, 5, 100);  // 近似 Top-5

db.Save("index.mdbv");                                  // 持久化
db.Load("index.mdbv");                                  // 恢复
```

## 路线图

- [x] **HNSW 索引**：分层可导航小世界图，近似检索已实现（M=16 / ef 可调 / 启发式选边）
- [x] **HNSW 增量维护**：节点级删除/更新，重连邻居保持连通，替代全量重建
- [x] **索引持久化**：格式 v3 含索引段（校验失败降级懒重建），冷启动免重建
- [ ] **pybind11 绑定**：提供 Python API，供 RAG 流水线直接调用
- [ ] **混合检索**：结合 BM25 关键词检索，文本召回更稳
- [ ] **RAG 示例**：金融文档切片 → embedding → 本库检索 → 拼接到 LLM 提示词
- [ ] **mmap 持久化**：大文件免加载全部进内存，支持冷启动即查
- [ ] **基准数据**：与 faiss / hnswlib 对比召回率与延迟，产出简历可用数据
