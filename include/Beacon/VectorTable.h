// 定长向量表：扁平连续存储（SlotCount() * Dim() 个 float），按 id 访问
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Beacon {

enum class Metric
{
    Cosine,        // 余弦相似度（文本 embedding 的默认选择）
    InnerProduct   // 内积（向量已归一化时等价于余弦）
};

struct Hit
{
    std::size_t Id;
    float Score;
};

class VectorTable
{
public:
    VectorTable() = default;
    VectorTable(std::size_t dim, Metric metric);

    std::size_t Add(const std::vector<float>& vec, const std::string& meta = {});
    bool Delete(std::size_t id);
    bool Update(std::size_t id, const std::vector<float>& vec, const std::string& meta = {});
    void Reserve(std::size_t slotCount);
    std::size_t Count() const;        // 存活向量数
    std::size_t SlotCount() const;    // 总槽位数（含 tombstone）
    std::size_t Dim() const;
    Metric GetMetric() const;

    bool Deleted(std::size_t id) const;
    const float* Vector(std::size_t id) const;
    const std::string& Metadata(std::size_t id) const;
    const float* FlatVectors() const;
    void SetData(std::size_t newDim, Metric newMetric, std::vector<float> newData, std::vector<std::string> newMetadata, std::vector<std::uint8_t> newDeletedFlags);

private:
    std::size_t dim_ = 0;
    Metric metric_ = Metric::Cosine;
    // data[id * dim, +dim)
    std::vector<float> data_;
    std::vector<std::string> metadata_;
    std::vector<std::uint8_t> deletedFlags_;
    std::size_t liveCount_ = 0;
};

}  // namespace Beacon
