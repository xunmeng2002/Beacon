// 向量数据库门面：向量表 + 暴力 top-K 检索 + 二进制持久化
#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "mdbvec/VectorTable.h"

namespace mdbvec {

struct Hit
{
    std::size_t id;
    float score;
};

class VectorDb
{
public:
    VectorDb() = default;
    VectorDb(std::size_t dim, Metric metric);

    std::size_t Add(const std::vector<float>& vec, const std::string& meta = {});
    std::vector<Hit> Search(const std::vector<float>& query, std::size_t k) const;

    std::size_t count() const;
    std::size_t dim() const;
    const std::string& metadata(std::size_t id) const;

    bool Save(const std::string& path) const;
    bool Load(const std::string& path);
    void Clear();

private:
    VectorTable table_;
};

}  // namespace mdbvec
