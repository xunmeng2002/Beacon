// 向量数据库门面：向量表 + 暴力 top-K 检索 + 可选 HNSW 索引 + 二进制持久化
#pragma once
#include "MdbVector/HnswIndex.h"

#include "MdbVector/VectorTable.h"

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace mdbvec {

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
    std::vector<Hit> Search(const std::vector<float>& query, std::size_t k) const;

    // 按预期最大条数预分配向量表存储，批量导入前调用避免重复扩容
    void Reserve(std::size_t slot_count);

    // HNSW 索引：EnableIndex 全量构建；Delete/Update 后懒重建
    void EnableIndex(std::size_t m = 16, std::size_t ef_construction = 200);
    void DisableIndex();
    bool IndexEnabled() const;
    std::vector<Hit> SearchIndexed(const std::vector<float>& query, std::size_t k, std::size_t ef = 100) const;

    std::size_t count() const;
    std::size_t dim() const;
    bool deleted(std::size_t id) const;
    std::string metadata(std::size_t id) const;   // 按值：const& 会逃逸锁，并发 Add/Update 下悬垂

    bool Save(const std::string& path) const;
    bool Load(const std::string& path);
    void Clear();

private:
    VectorTable table_;
    std::unique_ptr<HnswIndex> index_;
    mutable bool index_dirty_ = false;
    mutable std::shared_mutex rw_mutex_;   // 门面读写锁：所有公有方法均须持有，读共享/写独占
};

}  // namespace mdbvec
