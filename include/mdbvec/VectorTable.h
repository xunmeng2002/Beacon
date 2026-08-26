// 定长向量表：扁平连续存储（slot_count * dim），按 id 访问
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mdbvec {

enum class Metric
{
    kCosine,        // 余弦相似度（文本 embedding 的默认选择）
    kInnerProduct   // 内积（向量已归一化时等价于余弦）
};

struct Hit
{
    std::size_t id;
    float score;
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
    void Reserve(std::size_t slot_count);

    std::size_t count() const;        // 存活向量数
    std::size_t slot_count() const;   // 总槽位数（含 tombstone）
    std::size_t dim() const;
    Metric metric() const;

    bool deleted(std::size_t id) const;
    const float* vector(std::size_t id) const;
    const std::string& metadata(std::size_t id) const;

    // 扁平数据指针（供批量读写使用）
    const float* data() const;

    // 以已就绪数据构建（反序列化用）
    void set_data(std::size_t dim, Metric metric,
                  std::vector<float> data, std::vector<std::string> metadata,
                  std::vector<std::uint8_t> deleted);

private:
    std::size_t dim_ = 0;
    Metric metric_ = Metric::kCosine;
    // data_[id * dim_, +dim_)
    std::vector<float> data_;
    std::vector<std::string> metadata_;
    std::vector<std::uint8_t> deleted_;
    std::size_t live_count_ = 0;
};

}  // namespace mdbvec
