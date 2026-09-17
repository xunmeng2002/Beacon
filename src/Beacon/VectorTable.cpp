#include "Beacon/VectorTable.h"

#include "Beacon/Metrics.h"

#include <algorithm>
#include <utility>

namespace Beacon {

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
    if (metric_ == Metric::Cosine)
    {
        L2Normalize(stored.data(), dim_);
    }
    data_.insert(data_.end(), stored.begin(), stored.end());
    metadata_.push_back(meta);
    deletedFlags_.push_back(0);
    ++liveCount_;
    return metadata_.size() - 1;
}
void VectorTable::Reserve(std::size_t slotCount)
{
    const std::size_t floatCount = slotCount * dim_;
    // 溢出保护：dim_ 为 0 或乘积回绕时放弃预分配，Add 仍会按需扩容
    if (dim_ == 0 || floatCount / dim_ != slotCount)
    {
        return;
    }
    data_.reserve(floatCount);
    metadata_.reserve(slotCount);
    deletedFlags_.reserve(slotCount);
}
bool VectorTable::Delete(std::size_t id)
{
    if (id >= metadata_.size() || deletedFlags_[id] != 0)
    {
        return false;
    }
    deletedFlags_[id] = 1;
    --liveCount_;
    return true;
}
bool VectorTable::Update(std::size_t id, const std::vector<float>& vec, const std::string& meta)
{
    if (id >= metadata_.size() || vec.size() != dim_)
    {
        return false;
    }
    std::vector<float> stored = vec;
    if (metric_ == Metric::Cosine)
    {
        L2Normalize(stored.data(), dim_);
    }
    float* dst = data_.data() + id * dim_;
    std::copy(stored.begin(), stored.end(), dst);
    metadata_[id] = meta;
    if (deletedFlags_[id] != 0)
    {
        deletedFlags_[id] = 0;
        ++liveCount_;
    }
    return true;
}
std::size_t VectorTable::Count() const
{
    return liveCount_;
}
std::size_t VectorTable::SlotCount() const
{
    return metadata_.size();
}
std::size_t VectorTable::Dim() const
{
    return dim_;
}
Metric VectorTable::GetMetric() const
{
    return metric_;
}
bool VectorTable::Deleted(std::size_t id) const
{
    return deletedFlags_[id] != 0;
}
const float* VectorTable::Vector(std::size_t id) const
{
    return data_.data() + id * dim_;
}
const float* VectorTable::FlatVectors() const
{
    return data_.data();
}
const std::string& VectorTable::Metadata(std::size_t id) const
{
    return metadata_[id];
}
void VectorTable::SetData(std::size_t newDim, Metric newMetric, std::vector<float> newData, std::vector<std::string> newMetadata_, std::vector<std::uint8_t> newDeletedFlags)
{
    dim_ = newDim;
    metric_ = newMetric;
    data_ = std::move(newData);
    metadata_ = std::move(newMetadata_);
    deletedFlags_ = std::move(newDeletedFlags);
    liveCount_ = metadata_.size();
    for (std::uint8_t flag : deletedFlags_)
    {
        if (flag != 0)
        {
            --liveCount_;
        }
    }
}

}  // namespace Beacon
