#include "Beacon/VectorDb.h"

#include "Beacon/Metrics.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <queue>

namespace Beacon {

namespace
{
constexpr char Magic[4] = { 'B', 'E', 'A', 'C' };
// v1 全存活；v2 起含 tombstone 段；v3 起含 HNSW 索引段（可缺失，缺失即懒重建）
constexpr std::uint32_t FormatVersion = 3;

// 优先队列按 score 最小堆组织（堆顶是最差候选）
struct MinScoreFirst
{
    bool operator()(const Hit& a, const Hit& b) const
    {
        return a.Score > b.Score;
    }
};

struct FileHeader
{
    std::uint32_t Version;
    std::uint64_t Dim;
    std::uint8_t MetricCode;
    std::uint64_t SlotCount;
};

std::uint8_t MetricToU8(Metric metric)
{
    return metric == Metric::Cosine ? 0u : 1u;
}

Metric U8ToMetric(std::uint8_t metricCode)
{
    return metricCode == 0 ? Metric::Cosine : Metric::InnerProduct;
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
    const std::uint64_t floatCount =
        static_cast<std::uint64_t>(table.Dim()) * static_cast<std::uint64_t>(table.SlotCount());
    out.write(reinterpret_cast<const char*>(table.FlatVectors()),
              static_cast<std::streamsize>(floatCount * sizeof(float)));
    return static_cast<bool>(out);
}

bool ReadVectorData(std::ifstream& in, std::size_t dim, std::size_t slotCount,
                    std::vector<float>& data)
{
    data.resize(dim * slotCount);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(in);
}

bool WriteTombstones(std::ofstream& out, const VectorTable& table)
{
    for (std::size_t id = 0; id < table.SlotCount(); ++id)
    {
        const std::uint8_t flag = table.Deleted(id) ? 1u : 0u;
        out.write(reinterpret_cast<const char*>(&flag), sizeof(flag));
    }
    return static_cast<bool>(out);
}

bool ReadTombstones(std::ifstream& in, std::size_t slotCount, std::vector<std::uint8_t>& deleted)
{
    deleted.resize(slotCount);
    in.read(reinterpret_cast<char*>(deleted.data()),
            static_cast<std::streamsize>(deleted.size()));
    return static_cast<bool>(in);
}

bool WriteMetadata(std::ofstream& out, const VectorTable& table)
{
    for (std::size_t id = 0; id < table.SlotCount(); ++id)
    {
        if (!WriteStr(out, table.Metadata(id)))
        {
            return false;
        }
    }
    return true;
}

bool ReadMetadata(std::ifstream& in, std::size_t slotCount, std::vector<std::string>& metadata)
{
    metadata.reserve(slotCount);
    for (std::size_t i = 0; i < slotCount; ++i)
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
    if (!in || std::string(magic, 4) != std::string(Magic, 4))
    {
        return false;
    }

    in.read(reinterpret_cast<char*>(&header.Version), sizeof(header.Version));
    in.read(reinterpret_cast<char*>(&header.Dim), sizeof(header.Dim));
    in.read(reinterpret_cast<char*>(&header.MetricCode), sizeof(header.MetricCode));
    in.read(reinterpret_cast<char*>(&header.SlotCount), sizeof(header.SlotCount));
    if (!in || header.Version < 1 || header.Version > FormatVersion)
    {
        return false;
    }
    return true;
}

bool ReadPayload(std::ifstream& in, const FileHeader& header,
                 std::vector<float>& data, std::vector<std::uint8_t>& deleted,
                 std::vector<std::string>& metadata)
{
    const auto dim = static_cast<std::size_t>(header.Dim);
    const auto slotCount = static_cast<std::size_t>(header.SlotCount);
    if (!ReadVectorData(in, dim, slotCount, data))
    {
        return false;
    }
    if (header.Version >= 2)
    {
        if (!ReadTombstones(in, slotCount, deleted))
        {
            return false;
        }
    }
    else
    {
        deleted.assign(slotCount, 0u);
    }
    return ReadMetadata(in, slotCount, metadata);
}

// 全扫描 + 最小堆选出 top-K，返回按分数降序的命中列表；跳过已删除向量
std::vector<Hit> SelectTopK(const float* query, const VectorTable& table, std::size_t k)
{
    std::priority_queue<Hit, std::vector<Hit>, MinScoreFirst> heap;
    const std::size_t dim = table.Dim();
    for (std::size_t id = 0; id < table.SlotCount(); ++id)
    {
        if (table.Deleted(id))
        {
            continue;
        }
        const float score = DotProduct(query, table.Vector(id), dim);
        if (heap.size() < k)
        {
            heap.push(Hit{ id, score });
        }
        else if (score > heap.top().Score)
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

VectorDb::VectorDb(std::size_t dim, Metric metric) : table(dim, metric)
{
}

void VectorDb::Reserve(std::size_t slotCount)
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    table.Reserve(slotCount);
}

std::size_t VectorDb::Add(const std::vector<float>& vec, const std::string& meta)
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    const std::size_t id = table.Add(vec, meta);
    if (id != static_cast<std::size_t>(-1) && index && !indexDirty)
    {
        index->Add(id);
    }
    return id;
}

bool VectorDb::Update(std::size_t id, const std::vector<float>& vec, const std::string& meta)
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    const bool ok = table.Update(id, vec, meta);
    if (ok && index && !indexDirty)
    {
        index->Add(id);   // 幂等：在图中则先移除再按新向量重插
    }
    return ok;
}

bool VectorDb::Delete(std::size_t id)
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    const bool ok = table.Delete(id);
    if (ok && index && !indexDirty)
    {
        index->Remove(id);   // 节点级删除，不再触发全量重建
    }
    return ok;
}

std::vector<Hit> VectorDb::SearchExact(const std::vector<float>& query, std::size_t k) const
{
    std::shared_lock<std::shared_mutex> lock(rwMutex);
    std::vector<Hit> result;
    const std::size_t n = table.Count();
    if (n == 0 || k == 0 || query.size() != table.Dim())
    {
        return result;
    }

    std::vector<float> q = query;
    if (table.GetMetric() == Metric::Cosine)
    {
        L2Normalize(q.data(), q.size());
    }

    return SelectTopK(q.data(), table, std::min(k, n));
}

void VectorDb::EnableIndex(std::size_t m, std::size_t efConstruction)
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    index = std::make_unique<HnswIndex>(&table, m, efConstruction);
    index->Rebuild();
    indexDirty = false;
}

void VectorDb::DisableIndex()
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    index = nullptr;
    indexDirty = false;
}

bool VectorDb::IndexEnabled() const
{
    std::shared_lock<std::shared_mutex> lock(rwMutex);
    return index != nullptr;
}

std::vector<Hit> VectorDb::SearchIndexed(const std::vector<float>& query, std::size_t k, std::size_t ef) const
{
    // 快路径：索引干净 → 共享锁，读读并行；脏/无索引再落独占锁重建
    // 注意：shared_mutex 无写者优先，持续读负载可能饿写者（学习项目可接受）
    {
        std::shared_lock<std::shared_mutex> lock(rwMutex);
        if (!index)
        {
            return {};
        }
        if (!indexDirty)
        {
            return index->Search(query, k, ef);
        }
    }
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    if (!index)
    {
        return {};   // 防共享锁释放后 DisableIndex 竞态，独占锁下重查
    }
    if (indexDirty)
    {
        index->Rebuild();
        indexDirty = false;
    }
    return index->Search(query, k, ef);
}

std::size_t VectorDb::Count() const
{
    std::shared_lock<std::shared_mutex> lock(rwMutex);
    return table.Count();
}

std::size_t VectorDb::Dim() const
{
    std::shared_lock<std::shared_mutex> lock(rwMutex);
    return table.Dim();
}

bool VectorDb::Deleted(std::size_t id) const
{
    std::shared_lock<std::shared_mutex> lock(rwMutex);
    return id < table.SlotCount() && table.Deleted(id);   // 越界 id 视为不存在
}

std::string VectorDb::Metadata(std::size_t id) const
{
    std::shared_lock<std::shared_mutex> lock(rwMutex);
    // 按值返回：引用会逃逸锁，并发 Add/Update 下悬垂；越界 id 返回空串
    return id < table.SlotCount() ? table.Metadata(id) : std::string{};
}

void VectorDb::Clear()
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    table = VectorTable();
    index = nullptr;
    indexDirty = false;
}

bool VectorDb::Save(const std::string& path) const
{
    // 独占锁：串行化并发 Save 到同一路径（文件级竞态），并保证迭代 DB 状态期间无写者
    std::unique_lock<std::shared_mutex> lock(rwMutex);
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    const auto dim = static_cast<std::uint64_t>(table.Dim());
    const auto slotCount = static_cast<std::uint64_t>(table.SlotCount());
    const auto metric = MetricToU8(table.GetMetric());

    out.write(Magic, 4);
    out.write(reinterpret_cast<const char*>(&FormatVersion), sizeof(FormatVersion));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    out.write(reinterpret_cast<const char*>(&slotCount), sizeof(slotCount));
    if (!out)
    {
        return false;
    }
    if (slotCount > 0 && !(WriteVectorData(out, table) && WriteTombstones(out, table) && WriteMetadata(out, table)))
    {
        return false;
    }
    // v3 起：向量段权威，索引段为可校验缓存（缺失/损坏即降级懒重建）
    const std::uint8_t hasIndex = (index && !indexDirty) ? 1u : 0u;
    out.write(reinterpret_cast<const char*>(&hasIndex), sizeof(hasIndex));
    if (hasIndex && !index->Write(out))
    {
        return false;
    }
    return static_cast<bool>(out);
}

bool VectorDb::Load(const std::string& path)
{
    std::unique_lock<std::shared_mutex> lock(rwMutex);
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
    if (header.SlotCount > 0)
    {
        if (!ReadPayload(in, header, data, deleted, metadata))
        {
            return false;
        }
    }

    const std::size_t oldDim = table.Dim();
    const Metric oldMetric = table.GetMetric();
    table.SetData(static_cast<std::size_t>(header.Dim), U8ToMetric(header.MetricCode), std::move(data), std::move(metadata), std::move(deleted));
    if (index)
    {
        // 维度/度量变更时，旧索引的 dim/metric 是旧表快照，按错维打分 → 丢弃重建
        if (oldDim != static_cast<std::size_t>(header.Dim) || oldMetric != U8ToMetric(header.MetricCode))
        {
            index.reset();
        }
        else
        {
            indexDirty = true;
        }
    }
    if (header.Version >= 3)
    {
        std::uint8_t hasIndex = 0;
        in.read(reinterpret_cast<char*>(&hasIndex), sizeof(hasIndex));
        if (hasIndex)
        {
            if (!index)
            {
                index = std::make_unique<HnswIndex>(&table);
            }
            // 索引段校验失败/截断 → 标记脏，下次 SearchIndexed 懒重建（向量数据不受影响）
            indexDirty = !index->Read(in);
        }
    }
    return true;
}

}  // namespace Beacon
