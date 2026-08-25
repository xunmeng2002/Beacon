#include "mdbvec/VectorDb.h"

#include "mdbvec/Metrics.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <queue>

namespace mdbvec {

namespace
{
constexpr char kMagic[4] = { 'M', 'D', 'B', 'V' };
// v1 全存活；v2 起含 tombstone 段；v3 起含 HNSW 索引段（可缺失，缺失即懒重建）
constexpr std::uint32_t kFormatVersion = 3;

// 优先队列按 score 最小堆组织（堆顶是最差候选）
struct MinScoreFirst
{
    bool operator()(const Hit& a, const Hit& b) const
    {
        return a.score > b.score;
    }
};

struct FileHeader
{
    std::uint32_t version;
    std::uint64_t dim;
    std::uint8_t metric;
    std::uint64_t slot_count;
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
        static_cast<std::uint64_t>(table.dim()) * static_cast<std::uint64_t>(table.slot_count());
    out.write(reinterpret_cast<const char*>(table.data()),
              static_cast<std::streamsize>(float_count * sizeof(float)));
    return static_cast<bool>(out);
}

bool ReadVectorData(std::ifstream& in, std::size_t dim, std::size_t slot_count,
                    std::vector<float>& data)
{
    data.resize(dim * slot_count);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(in);
}

bool WriteTombstones(std::ofstream& out, const VectorTable& table)
{
    for (std::size_t id = 0; id < table.slot_count(); ++id)
    {
        const std::uint8_t flag = table.deleted(id) ? 1u : 0u;
        out.write(reinterpret_cast<const char*>(&flag), sizeof(flag));
    }
    return static_cast<bool>(out);
}

bool ReadTombstones(std::ifstream& in, std::size_t slot_count, std::vector<std::uint8_t>& deleted)
{
    deleted.resize(slot_count);
    in.read(reinterpret_cast<char*>(deleted.data()),
            static_cast<std::streamsize>(deleted.size()));
    return static_cast<bool>(in);
}

bool WriteMetadata(std::ofstream& out, const VectorTable& table)
{
    for (std::size_t id = 0; id < table.slot_count(); ++id)
    {
        if (!WriteStr(out, table.metadata(id)))
        {
            return false;
        }
    }
    return true;
}

bool ReadMetadata(std::ifstream& in, std::size_t slot_count, std::vector<std::string>& metadata)
{
    metadata.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i)
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

bool ReadHeader(std::ifstream& in, FileHeader& header)
{
    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != std::string(kMagic, 4))
    {
        return false;
    }

    in.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    in.read(reinterpret_cast<char*>(&header.dim), sizeof(header.dim));
    in.read(reinterpret_cast<char*>(&header.metric), sizeof(header.metric));
    in.read(reinterpret_cast<char*>(&header.slot_count), sizeof(header.slot_count));
    if (!in || header.version < 1 || header.version > kFormatVersion)
    {
        return false;
    }
    return true;
}

bool ReadPayload(std::ifstream& in, const FileHeader& header,
                 std::vector<float>& data, std::vector<std::uint8_t>& deleted,
                 std::vector<std::string>& metadata)
{
    const auto dim = static_cast<std::size_t>(header.dim);
    const auto slot_count = static_cast<std::size_t>(header.slot_count);
    if (!ReadVectorData(in, dim, slot_count, data))
    {
        return false;
    }
    if (header.version >= 2)
    {
        if (!ReadTombstones(in, slot_count, deleted))
        {
            return false;
        }
    }
    else
    {
        deleted.assign(slot_count, 0u);
    }
    return ReadMetadata(in, slot_count, metadata);
}

// 全扫描 + 最小堆选出 top-K，返回按分数降序的命中列表；跳过已删除向量
std::vector<Hit> SelectTopK(const float* query, const VectorTable& table, std::size_t k)
{
    std::priority_queue<Hit, std::vector<Hit>, MinScoreFirst> heap;
    const std::size_t dim = table.dim();
    for (std::size_t id = 0; id < table.slot_count(); ++id)
    {
        if (table.deleted(id))
        {
            continue;
        }
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
    const std::size_t id = table_.Add(vec, meta);
    if (id != static_cast<std::size_t>(-1) && index_ && !index_dirty_)
    {
        index_->Add(id);
    }
    return id;
}

bool VectorDb::Update(std::size_t id, const std::vector<float>& vec, const std::string& meta)
{
    const bool ok = table_.Update(id, vec, meta);
    if (ok && index_)
    {
        index_->Add(id);   // 幂等：在图中则先移除再按新向量重插
    }
    return ok;
}

bool VectorDb::Delete(std::size_t id)
{
    const bool ok = table_.Delete(id);
    if (ok && index_)
    {
        index_->Remove(id);   // 节点级删除，不再触发全量重建
    }
    return ok;
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

void VectorDb::EnableIndex(std::size_t m, std::size_t ef_construction)
{
    index_ = std::make_unique<HnswIndex>(&table_, m, ef_construction);
    index_->Rebuild();
    index_dirty_ = false;
}

void VectorDb::DisableIndex()
{
    index_ = nullptr;
    index_dirty_ = false;
}

bool VectorDb::IndexEnabled() const
{
    return index_ != nullptr;
}

std::vector<Hit> VectorDb::SearchIndexed(const std::vector<float>& query, std::size_t k,
                                         std::size_t ef) const
{
    if (!index_)
    {
        return {};
    }
    if (index_dirty_)
    {
        index_->Rebuild();
        index_dirty_ = false;
    }
    return index_->Search(query, k, ef);
}

std::size_t VectorDb::count() const
{
    return table_.count();
}

std::size_t VectorDb::dim() const
{
    return table_.dim();
}

bool VectorDb::deleted(std::size_t id) const
{
    return table_.deleted(id);
}

const std::string& VectorDb::metadata(std::size_t id) const
{
    return table_.metadata(id);
}

void VectorDb::Clear()
{
    table_ = VectorTable();
    index_ = nullptr;
    index_dirty_ = false;
}

bool VectorDb::Save(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    const auto dim = static_cast<std::uint64_t>(table_.dim());
    const auto slot_count = static_cast<std::uint64_t>(table_.slot_count());
    const auto metric = MetricToU8(table_.metric());

    out.write(kMagic, 4);
    out.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    out.write(reinterpret_cast<const char*>(&slot_count), sizeof(slot_count));
    if (!out)
    {
        return false;
    }

    if (slot_count > 0 &&
        !(WriteVectorData(out, table_) && WriteTombstones(out, table_) &&
          WriteMetadata(out, table_)))
    {
        return false;
    }
    // v3 起：向量段权威，索引段为可校验缓存（缺失/损坏即降级懒重建）
    const std::uint8_t has_index = index_ ? 1u : 0u;
    out.write(reinterpret_cast<const char*>(&has_index), sizeof(has_index));
    if (index_ && !index_->Write(out))
    {
        return false;
    }
    return static_cast<bool>(out);
}

bool VectorDb::Load(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }

    FileHeader header;
    if (!ReadHeader(in, header))
    {
        return false;
    }

    std::vector<float> data;
    std::vector<std::uint8_t> deleted;
    std::vector<std::string> metadata;
    if (header.slot_count > 0)
    {
        if (!ReadPayload(in, header, data, deleted, metadata))
        {
            return false;
        }
    }

    table_.set_data(static_cast<std::size_t>(header.dim), U8ToMetric(header.metric),
                    std::move(data), std::move(metadata), std::move(deleted));
    if (index_)
    {
        index_dirty_ = true;
    }
    if (header.version >= 3)
    {
        std::uint8_t has_index = 0;
        in.read(reinterpret_cast<char*>(&has_index), sizeof(has_index));
        if (has_index)
        {
            if (!index_)
            {
                index_ = std::make_unique<HnswIndex>(&table_);
            }
            // 索引段校验失败/截断 → 标记脏，下次 SearchIndexed 懒重建（向量数据不受影响）
            index_dirty_ = !index_->Read(in);
        }
    }
    return true;
}

}  // namespace mdbvec
