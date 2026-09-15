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

    // 追加一个向量，返回其 id；维度不匹配时返回 (size_t)-1
    std::size_t Add(const std::vector<float>& vec, const std::string& meta = {});

    // 软删除：置 tombstone，不搬数据，id 保持稳定
    bool Delete(std::size_t id);

    // 就地覆盖向量与元数据；对已删除 id 执行则复活
    bool Update(std::size_t id, const std::vector<float>& vec, const std::string& meta = {});

    // 按预期最大槽位数预分配连续存储，避免批量导入时多次扩容重排
    void Reserve(std::size_t slotCount);

    std::size_t Count() const;        // 存活向量数
    std::size_t SlotCount() const;    // 总槽位数（含 tombstone）
    std::size_t Dim() const;
    Metric GetMetric() const;

    // 前置条件：id < SlotCount()（门面 VectorDb 已在越界前拦截，内部调用均满足；为免热路径分支不加检查）
    bool Deleted(std::size_t id) const;
    const float* Vector(std::size_t id) const;
    const std::string& Metadata(std::size_t id) const;

    // 扁平向量缓冲首地址：SlotCount() * Dim() 个 float 连续存放，供整段序列化读写；单个向量用 Vector(id)
    const float* FlatVectors() const;

    // 以已就绪数据构建（反序列化用）
    void SetData(std::size_t newDim, Metric newMetric,
                 std::vector<float> newData, std::vector<std::string> newMetadata,
                 std::vector<std::uint8_t> newDeletedFlags);

private:
    std::size_t dim = 0;
    Metric metric = Metric::Cosine;
    // data[id * dim, +dim)
    std::vector<float> data;
    std::vector<std::string> metadata;
    std::vector<std::uint8_t> deletedFlags;
    std::size_t liveCount = 0;
};

}  // namespace Beacon
