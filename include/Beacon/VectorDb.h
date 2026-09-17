// 向量数据库门面：向量表 + 暴力 top-K 检索 + 可选 HNSW 索引 + 二进制持久化
#pragma once
#include "Beacon/HnswIndex.h"

#include "Beacon/VectorTable.h"

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace Beacon {

class VectorDb
{
public:
    VectorDb() = default;
    VectorDb(std::size_t dim, Metric metric);
    VectorDb(const VectorDb&) = delete;             // 门面含读写锁，不可拷贝/移动
    VectorDb& operator=(const VectorDb&) = delete;

    std::size_t Add(const std::vector<float>& vec, const std::string& meta = {});
    bool Update(std::size_t id, const std::vector<float>& vec, const std::string& meta = {});
    bool Delete(std::size_t id);
    // 精确 Top-K：全扫描打分，结果保证正确，不依赖索引
    std::vector<Hit> SearchExact(const std::vector<float>& query, std::size_t k) const;

    // 按预期最大条数预分配向量表存储，批量导入前调用避免重复扩容
    void Reserve(std::size_t slotCount);

    // HNSW 索引：EnableIndex 全量构建；Delete/Update 后懒重建
    void EnableIndex(std::size_t m = 16, std::size_t efConstruction = 200);
    void DisableIndex();
    bool IndexEnabled() const;
    std::vector<Hit> SearchIndexed(const std::vector<float>& query, std::size_t k, std::size_t ef = 100) const;

    std::size_t Count() const;
    std::size_t Dim() const;
    bool Deleted(std::size_t id) const;
    std::string Metadata(std::size_t id) const;   // 按值：const& 会逃逸锁，并发 Add/Update 下悬垂

    bool Save(const std::string& path) const;
    bool Load(const std::string& path);
    void Clear();

private:
    VectorTable table_;
    std::unique_ptr<HnswIndex> index_;
    mutable bool indexDirty_ = false;
    mutable std::shared_mutex rwMutex_;     // 门面读写锁：所有公有方法均须持有，读共享/写独占
};

}  // namespace Beacon
