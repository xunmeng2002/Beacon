// 定长向量表：扁平连续存储（count * dim），按 id 访问
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace mdbvec {

enum class Metric {
  Cosine,        // 余弦相似度（文本 embedding 的默认选择）
  InnerProduct   // 内积（向量已归一化时等价于余弦）
};

class VectorTable {
public:
  VectorTable() = default;
  VectorTable(std::size_t dim, Metric metric);

  // 追加一个向量，返回其 id；维度不匹配时返回 (size_t)-1
  std::size_t add(const std::vector<float>& vec, const std::string& meta = {});

  std::size_t count() const;
  std::size_t dim() const;
  Metric metric() const;

  const float* vector(std::size_t id) const;
  const std::string& metadata(std::size_t id) const;

  // 扁平数据指针（供批量读写使用）
  const float* data() const { return data_.data(); }

  // 以已就绪数据构建（反序列化用）
  void set_data(std::size_t dim, Metric metric,
                std::vector<float> data, std::vector<std::string> metadata);

private:
  std::size_t dim_ = 0;
  Metric metric_ = Metric::Cosine;
  std::vector<float> data_;               // data_[id * dim_, +dim_)
  std::vector<std::string> metadata_;
};

}  // namespace mdbvec
