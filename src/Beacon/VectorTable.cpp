#include "Beacon/VectorTable.h"

#include "Beacon/Metrics.h"

#include <algorithm>
#include <utility>

namespace Beacon {

VectorTable::VectorTable(std::size_t dim, Metric metric) : dim(dim), metric(metric)
{
}

std::size_t VectorTable::Add(const std::vector<float>& vec, const std::string& meta)
{
    if (vec.size() != dim)
    {
        return static_cast<std::size_t>(-1);
    }
    std::vector<float> stored = vec;
    if (metric == Metric::Cosine)
    {
        L2Normalize(stored.data(), dim);
    }
    data.insert(data.end(), stored.begin(), stored.end());
    metadata.push_back(meta);
    deletedFlags.push_back(0);
    ++liveCount;
    return metadata.size() - 1;
}

void VectorTable::Reserve(std::size_t slotCount)
{
    const std::size_t floatCount = slotCount * dim;
    // 溢出保护：dim 为 0 或乘积回绕时放弃预分配，Add 仍会按需扩容
    if (dim == 0 || floatCount / dim != slotCount)
    {
        return;
    }
    data.reserve(floatCount);
    metadata.reserve(slotCount);
    deletedFlags.reserve(slotCount);
}

bool VectorTable::Delete(std::size_t id)
{
    if (id >= metadata.size() || deletedFlags[id] != 0)
    {
        return false;
    }
    deletedFlags[id] = 1;
    --liveCount;
    return true;
}

bool VectorTable::Update(std::size_t id, const std::vector<float>& vec, const std::string& meta)
{
    if (id >= metadata.size() || vec.size() != dim)
    {
        return false;
    }
    std::vector<float> stored = vec;
    if (metric == Metric::Cosine)
    {
        L2Normalize(stored.data(), dim);
    }
    float* dst = data.data() + id * dim;
    std::copy(stored.begin(), stored.end(), dst);
    metadata[id] = meta;
    if (deletedFlags[id] != 0)
    {
        deletedFlags[id] = 0;
        ++liveCount;
    }
    return true;
}

std::size_t VectorTable::Count() const
{
    return liveCount;
}

std::size_t VectorTable::SlotCount() const
{
    return metadata.size();
}

std::size_t VectorTable::Dim() const
{
    return dim;
}

Metric VectorTable::GetMetric() const
{
    return metric;
}

bool VectorTable::Deleted(std::size_t id) const
{
    return deletedFlags[id] != 0;
}

const float* VectorTable::Vector(std::size_t id) const
{
    return data.data() + id * dim;
}

const float* VectorTable::FlatVectors() const
{
    return data.data();
}

const std::string& VectorTable::Metadata(std::size_t id) const
{
    return metadata[id];
}

void VectorTable::SetData(std::size_t newDim, Metric newMetric,
                          std::vector<float> newData, std::vector<std::string> newMetadata,
                          std::vector<std::uint8_t> newDeletedFlags)
{
    dim = newDim;
    metric = newMetric;
    data = std::move(newData);
    metadata = std::move(newMetadata);
    deletedFlags = std::move(newDeletedFlags);
    liveCount = metadata.size();
    for (std::uint8_t flag : deletedFlags)
    {
        if (flag != 0)
        {
            --liveCount;
        }
    }
}

}  // namespace Beacon
