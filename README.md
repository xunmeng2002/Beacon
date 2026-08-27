# Beacon

自研的**内存向量数据库**（C++20）。为 RAG（检索增强生成）场景提供基于余弦相似度 / 内积的 Top-K 检索，并支持二进制持久化。作为个人项目，重点在于吃透向量索引与检索的底层原理，而非堆砌依赖。

## 特性

- **扁平连续存储**：向量按 `count × dim` 连续存放，缓存友好，`O(1)` 按 id 访问
- **两种度量**：余弦相似度（文本 embedding 默认）、内积
  - 余弦模式在**插入时** L2 归一化，检索时只需一次点积，避免每次查询重复算范数
- **SIMD 加速**：`x86_64` 上点积走 **AVX2**（8 路 float 并行），其余平台自动标量回退
- **暴力 Top-K**：小规模数据直接全扫描 + 最小堆维护 K 个最优，结果按分数降序返回
- **二进制持久化**：`magic("BEAC") + version + dim + metric + count + 数据 + 元数据`，加载时校验魔数与版本
- **元数据**：每个向量可携带任意字符串（文档 id、原文片段等）
- **CRUD**：软删除（tombstone，数据不动、id 稳定）+ 就地更新（对已删除 id 执行则复活）
- **并发支持**：`VectorDb` 门面 `shared_mutex` 读写锁——读读并行、读写/写写互斥；检索 visited 池为线程局部，多线程查询共享读锁安全
- **HNSW 索引**：分层可导航小世界图，近似检索（`ef` 可调）；Delete/Update 走节点级增量维护（重连邻居保连通），暴力检索保留作精确对照
- **持久化格式版本化**：v2 起记录 tombstone；v3 起含 HNSW 索引段（向量段权威、索引段为可校验缓存，损坏即降级懒重建）；向后兼容 v1/v2

## 构建

需要 **CMake ≥ 3.20**（使用 `--preset` 需 ≥ 3.21）和任一 C++20 编译器（MSVC / GCC / Clang）。

**Windows / MSVC（推荐）**——使用 CMakePresets，内置 Ninja + vcpkg 工具链（需 `VCPKG_ROOT` 环境变量）：

```bash
cmake --preset x64-Release          # 或 x64-Debug
cmake --build out/build/x64-Release --config Release
```

**通用配置**（无 vcpkg / 其他平台）：

```bash
cmake -S . -B build -D BEACON_ENABLE_BENCH=OFF -D BUILD_UNIT_TESTS=OFF    # 关闭对比基准与单元测试，库本体零第三方依赖
cmake --build build --config Release
```

> x86_64 平台自动追加 `/arch:AVX2`（MSVC）或 `-mavx2`（GCC/Clang）。
>
> hnswlib 对比基准（`beacon_bench`）与单元测试（`UnitTests`，GoogleTest）都需要 vcpkg toolchain（`vcpkg.json` 声明 `hnswlib`、`gtest` 依赖）。两者均关闭时纯标准库编译，库本体不引入任何第三方依赖。

## 运行 Demo

构建产物统一输出到 `bin/$<CONFIG>`（可执行文件）与 `lib/$<CONFIG>`（静态库），preset 与通用配置一致：

```bash
./bin/Release/TestBeacon.exe            # Windows（x64-Release preset 或普通配置构建）
./bin/Release/TestBeacon                # Linux/macOS（配置时 -D CMAKE_BUILD_TYPE=Release）
```

Demo 输出四类结果：

1. **3 维余弦检索 + 持久化往返**：验证归一化、Top-K 排序、软删除/更新、save/load 正确性
2. **10k × 64 维暴力检索延迟**：量测全扫描耗时（AVX2 下约 0.13 ms）
3. **30k × 64 维 HNSW vs 暴力**：recall@10 与延迟对比 + 增量删 3000 节点耗时（毫秒级）+ 含索引持久化重载免重建验证
4. **并发读写压力**：4 读者并发 `SearchIndexed` + 2 写者 Add/Update/Delete，断言最终 `count` 确定一致（与线程交错顺序无关）

## 运行单元测试

单元测试使用 GoogleTest，覆盖度量 / 向量表 / HNSW 索引 / 门面四个模块（`test/unittest/`）。`BUILD_UNIT_TESTS` 默认开启，全部用例通过时退出码为 0：

```bash
./bin/Release/UnitTests.exe            # Windows
./bin/Release/UnitTests                # Linux/macOS
```

## Python 绑定

门面 `VectorDb` 经 pybind11 导出为模块 `beacon`，向量入参支持 `list` 与 numpy 数组。默认关闭（`BEACON_ENABLE_PYTHON=OFF`），库本体保持零第三方依赖：

```bash
python -m pip install pybind11            # 一次性安装（PyPI）
cmake --preset x64-Release -D BEACON_ENABLE_PYTHON=ON
cmake --build out/build/x64-Release --config Release
```

产物 `bin/Release/beacon.cp311-win_amd64.pyd`（Windows；`cp311-win_amd64` 为 SOABI 后缀，随 Python 版本变化，导入名始终是 `beacon`），用构建时同一个 Python 导入：

```python
import sys
sys.path.insert(0, "bin/Release")      # 或把该目录加进 PYTHONPATH
import beacon

db = beacon.VectorDb(384, beacon.Metric.kCosine)
db.add([0.1, 0.2, ...], "文档A")         # list 或 numpy.float32 数组
db.enable_index()
hits = db.search_indexed(query, 5, ef=100)   # -> [Hit(id=..., score=...)]
db.save("index.beacon")
```

> **注意**：`.pyd` 绑定的是构建时 `find_package(Python3)` 找到的解释器（本机 `C:\Python\Python311`）；若 RAG 用虚拟环境，请用该环境的 Python 进行配置/构建。numpy 2.x 兼容（pybind11 ≥ 3.0）。

运行绑定测试（pytest，需 `pip install pytest`；也可直接运行脚本，如 VS 里点运行）：

```bash
python -m pytest test/python -v                 # pytest 方式，20 个用例
python test/python/test_python_bindings.py      # 直接运行方式
```

两种方式都覆盖：CRUD / 精确与近似检索 / 持久化 / 边界与异常。

## 目录结构

库本体零第三方依赖，构建产物输出到 `bin/$<CONFIG>` / `lib/$<CONFIG>`：

```
include/Beacon/        对外 API（消费方唯一包含入口，安装即此目录）
  Metrics.h       距离度量：点积、L2 范数、L2 归一化
  VectorTable.h   定长向量表 + 公共类型 Metric / Hit
  HnswIndex.h     HNSW 分层小世界图索引（近似检索）
  VectorDb.h      门面：精确/近似搜索、持久化、清空（主入口）
src/Beacon/            对内实现（CMake 仅 PUBLIC 暴露 include/，此处不进消费方包含路径）
  Metrics.cpp     AVX2 / 标量双路径点积
  VectorTable.cpp
  HnswIndex.cpp   HNSW 建图/检索
  VectorDb.cpp    最小堆 Top-K、Save/Load
test/TestBeacon/     Demo 可执行文件源码（TestBeacon.cpp，四类演示）
test/unittest/          GoogleTest 单元测试（Metrics / VectorTable / HnswIndex / VectorDb，需 vcpkg 提供 gtest）
test/python/            Python 绑定 pytest 测试（test_python_bindings.py，可直接运行也可 pytest）
bindings/python/        pybind11 绑定模块 beacon（需 pip install pybind11，BEACON_ENABLE_PYTHON=ON 开启）
bench/                  与 hnswlib 的对比基准（需 vcpkg，默认开启，可 BEACON_ENABLE_BENCH=OFF 关闭）
bin/  lib/              构建产物（按 $<CONFIG> 分目录：Release/Debug）
```

## 快速上手

```cpp
#include "Beacon/VectorDb.h"

beacon::VectorDb db(384, beacon::Metric::kCosine);
db.Add({ 0.1f, 0.2f, /* ... */ }, "文档A");

db.Update(id, { 0.2f, 0.1f, /* ... */ }, "文档A(修订)");  // 就地更新
db.Delete(id);                                            // 软删除，id 仍有效

auto hits = db.SearchExact({ 0.15f, 0.21f, /* ... */ }, 5);  // 精确 Top-5

db.EnableIndex();                                       // 构建 HNSW 索引
auto near = db.SearchIndexed({ 0.15f, 0.21f, /* ... */ }, 5, 100);  // 近似 Top-5

db.Save("index.beacon");                                  // 持久化
db.Load("index.beacon");                                  // 恢复
```

## 路线图

- [x] **HNSW 索引**：分层可导航小世界图，近似检索已实现（M=16 / ef 可调 / 启发式选边）
- [x] **HNSW 增量维护**：节点级删除/更新，重连邻居保持连通，替代全量重建
- [x] **索引持久化**：格式 v3 含索引段（校验失败降级懒重建），冷启动免重建
- [x] **pybind11 绑定**：提供 Python API，供 RAG 流水线直接调用
- [ ] **混合检索**：结合 BM25 关键词检索，文本召回更稳
- [ ] **RAG 示例**：金融文档切片 → embedding → 本库检索 → 拼接到 LLM 提示词
- [ ] **mmap 持久化**：大文件免加载全部进内存，支持冷启动即查
- [x] **基准数据**：与 hnswlib 对比召回率与延迟（`bench/bench_compare.cpp`；100k×128 下 recall 0.331 vs 0.301）
