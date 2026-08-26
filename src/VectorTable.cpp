#include "mdbvec/VectorTable.h"

#include "mdbvec/Metrics.h"

#include <algorithm>
#include <utility>

namespace mdbvec {

VectorTable::VectorTable(std::size_t dim, Metric metric) : dim_(dim), metric_(metric)
{
}

std::size_t VectorTable::Add(const std::vector<float>& vec, const std::string& meta)
{
    if (vec.size() != dim_)
    {
        return static_cast<std::size_t>(-1);
    }
    std::vector<float> stored = vec;
    if (metric_ == Metric::kCosine)
    {
        L2Normalize(stored.data(), dim_);
    }
    data_.insert(data_.end(), stored.begin(), stored.end());
    metadata_.push_back(meta);
    deleted_.push_back(0);
    ++live_count_;
    return metadata_.size() - 1;
}

void VectorTable::Reserve(std::size_t slot_count)
{
    const std::size_t float_count = slot_count * dim_;
    // 溢出保护：dim_ 为 0 或乘积回绕时放弃预分配，Add 仍会按需扩容
    if (dim_ == 0 || float_count / dim_ != slot_count)
    {
        return;
    }
    data_.reserve(float_count);
    metadata_.reserve(slot_count);
    deleted_.reserve(slot_count);
}

bool VectorTable::Delete(std::size_t id)
{
    if (id >= metadata_.size() || deleted_[id] != 0)
    {
        return false;
    }
    deleted_[id] = 1;
    --live_count_;
    return true;
}

bool VectorTable::Update(std::size_t id, const std::vector<float>& vec, const std::string& meta)
{
    if (id >= metadata_.size() || vec.size() != dim_)
    {
        return false;
    }
    std::vector<float> stored = vec;
    if (metric_ == Metric::kCosine)
    {
        L2Normalize(stored.data(), dim_);
    }
    float* dst = data_.data() + id * dim_;
    std::copy(stored.begin(), stored.end(), dst);
    metadata_[id] = meta;
    if (deleted_[id] != 0)
    {
        deleted_[id] = 0;
        ++live_count_;
    }
    return true;
}

std::size_t VectorTable::count() const
{
    return live_count_;
}

std::size_t VectorTable::slot_count() const
{
    return metadata_.size();
}

std::size_t VectorTable::dim() const
{
    return dim_;
}

Metric VectorTable::metric() const
{
    return metric_;
}

bool VectorTable::deleted(std::size_t id) const
{
    return deleted_[id] != 0;
}

const float* VectorTable::vector(std::size_t id) const
{
    return data_.data() + id * dim_;
}

const float* VectorTable::data() const
{
    return data_.data();
}

const std::string& VectorTable::metadata(std::size_t id) const
{
    return metadata_[id];
}

void VectorTable::set_data(std::size_t dim, Metric metric,
                           std::vector<float> data, std::vector<std::string> metadata,
                           std::vector<std::uint8_t> deleted)
{
    dim_ = dim;
    metric_ = metric;
    data_ = std::move(data);
    metadata_ = std::move(metadata);
    deleted_ = std::move(deleted);
    live_count_ = metadata_.size();
    for (std::uint8_t flag : deleted_)
    {
        if (flag != 0)
        {
            --live_count_;
        }
    }
}

}  // namespace mdbvec
