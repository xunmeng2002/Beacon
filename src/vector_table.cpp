#include "mdbvec/vector_table.h"

#include <utility>

#include "mdbvec/metrics.h"

namespace mdbvec {

VectorTable::VectorTable(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {}

std::size_t VectorTable::add(const std::vector<float>& vec, const std::string& meta) {
  if (vec.size() != dim_) {
    return static_cast<std::size_t>(-1);
  }
  std::vector<float> stored = vec;
  if (metric_ == Metric::Cosine) {
    l2_normalize(stored.data(), dim_);
  }
  data_.insert(data_.end(), stored.begin(), stored.end());
  metadata_.push_back(meta);
  return metadata_.size() - 1;
}

std::size_t VectorTable::count() const { return metadata_.size(); }
std::size_t VectorTable::dim() const { return dim_; }
Metric VectorTable::metric() const { return metric_; }

const float* VectorTable::vector(std::size_t id) const {
  return data_.data() + id * dim_;
}

const std::string& VectorTable::metadata(std::size_t id) const {
  return metadata_[id];
}

void VectorTable::set_data(std::size_t dim, Metric metric,
                           std::vector<float> data, std::vector<std::string> metadata) {
  dim_ = dim;
  metric_ = metric;
  data_ = std::move(data);
  metadata_ = std::move(metadata);
}

}  // namespace mdbvec
