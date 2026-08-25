#include "mdbvec/VectorDb.h"

#include "mdbvec/Metrics.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <queue>

namespace mdbvec {

namespace
{
constexpr char kMagic[4] = { 'M', 'D', 'B', 'V' };
constexpr std::uint32_t kFormatVersion = 1;

// 优先队列按 score 最小堆组织（堆顶是最差候选）
struct MinScoreFirst
{
    bool operator()(const Hit& a, const Hit& b) const
    {
        return a.score > b.score;
    }
};

std::uint8_t MetricToU8(Metric m)
{
    return m == Metric::kCosine ? 0u : 1u;
}

Metric U8ToMetric(std::uint8_t v)
{
    return v == 0 ? Metric::kCosine : Metric::kInnerProduct;
}

bool WriteStr(std::ofstream& out, const std::string& s)
{
    const auto len = static_cast<std::uint64_t>(s.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (!s.empty())
    {
        out.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
    return static_cast<bool>(out);
}

bool ReadStr(std::ifstream& in, std::string& s)
{
    std::uint64_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!in)
    {
        return false;
    }
    s.resize(static_cast<std::size_t>(len));
    if (len > 0)
    {
        in.read(s.data(), static_cast<std::streamsize>(s.size()));
    }
    return static_cast<bool>(in);
}

bool WriteVectorData(std::ofstream& out, const VectorTable& table)
{
    const std::uint64_t float_count =
        static_cast<std::uint64_t>(table.dim()) * static_cast<std::uint64_t>(table.count());
    out.write(reinterpret_cast<const char*>(table.data()),
              static_cast<std::streamsize>(float_count * sizeof(float)));
    return static_cast<bool>(out);
}

bool ReadVectorData(std::ifstream& in, std::size_t dim, std::size_t count,
                    std::vector<float>& data)
{
    data.resize(dim * count);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(in);
}

bool WriteMetadata(std::ofstream& out, const VectorTable& table)
{
    for (std::size_t id = 0; id < table.count(); ++id)
    {
        if (!WriteStr(out, table.metadata(id)))
        {
            return false;
        }
    }
    return true;
}

bool ReadMetadata(std::ifstream& in, std::size_t count, std::vector<std::string>& metadata)
{
    metadata.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        std::string s;
        if (!ReadStr(in, s))
        {
            return false;
        }
        metadata.push_back(std::move(s));
    }
    return true;
}

bool ReadHeader(std::ifstream& in, std::uint64_t& dim, std::uint8_t& metric, std::uint64_t& count)
{
    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != std::string(kMagic, 4))
    {
        return false;
    }

    std::uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&metric), sizeof(metric));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || version != kFormatVersion)
    {
        return false;
    }
    return true;
}

// 全扫描 + 最小堆选出 top-K，返回按分数降序的命中列表
std::vector<Hit> SelectTopK(const float* query, const VectorTable& table, std::size_t k)
{
    std::priority_queue<Hit, std::vector<Hit>, MinScoreFirst> heap;
    const std::size_t dim = table.dim();
    for (std::size_t id = 0; id < table.count(); ++id)
    {
        const float score = DotProduct(query, table.vector(id), dim);
        if (heap.size() < k)
        {
            heap.push(Hit{ id, score });
        }
        else if (score > heap.top().score)
        {
            heap.pop();
            heap.push(Hit{ id, score });
        }
    }

    std::vector<Hit> result;
    result.reserve(heap.size());
    while (!heap.empty())
    {
        result.push_back(heap.top());
        heap.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}
}  // namespace

VectorDb::VectorDb(std::size_t dim, Metric metric) : table_(dim, metric)
{
}

std::size_t VectorDb::Add(const std::vector<float>& vec, const std::string& meta)
{
    return table_.Add(vec, meta);
}

std::vector<Hit> VectorDb::Search(const std::vector<float>& query, std::size_t k) const
{
    std::vector<Hit> result;
    const std::size_t n = table_.count();
    if (n == 0 || k == 0 || query.size() != table_.dim())
    {
        return result;
    }

    std::vector<float> q = query;
    if (table_.metric() == Metric::kCosine)
    {
        L2Normalize(q.data(), q.size());
    }

    return SelectTopK(q.data(), table_, std::min(k, n));
}

std::size_t VectorDb::count() const
{
    return table_.count();
}

std::size_t VectorDb::dim() const
{
    return table_.dim();
}

const std::string& VectorDb::metadata(std::size_t id) const
{
    return table_.metadata(id);
}

void VectorDb::Clear()
{
    table_ = VectorTable();
}

bool VectorDb::Save(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    const auto dim = static_cast<std::uint64_t>(table_.dim());
    const auto count = static_cast<std::uint64_t>(table_.count());
    const auto metric = MetricToU8(table_.metric());

    out.write(kMagic, 4);
    out.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!out)
    {
        return false;
    }

    if (count > 0)
    {
        return WriteVectorData(out, table_) && WriteMetadata(out, table_);
    }
    return true;
}

bool VectorDb::Load(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }

    std::uint64_t dim = 0;
    std::uint64_t count = 0;
    std::uint8_t metric = 0;
    if (!ReadHeader(in, dim, metric, count))
    {
        return false;
    }

    std::vector<float> data;
    std::vector<std::string> metadata;
    if (count > 0)
    {
        if (!ReadVectorData(in, static_cast<std::size_t>(dim),
                            static_cast<std::size_t>(count), data))
        {
            return false;
        }
        if (!ReadMetadata(in, static_cast<std::size_t>(count), metadata))
        {
            return false;
        }
    }

    table_.set_data(static_cast<std::size_t>(dim), U8ToMetric(metric),
                    std::move(data), std::move(metadata));
    return true;
}

}  // namespace mdbvec
