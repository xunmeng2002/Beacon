#include "Beacon/HnswIndex.h"

#include "Beacon/Metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <queue>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Beacon {

HnswIndex::HnswIndex(const VectorTable* table, std::size_t m, std::size_t efConstruction)
    : table(table),
      dim(table->Dim()),
      metric(table->GetMetric()),
      maxNeighbors(std::max<std::size_t>(m, 3)),
      efConstruction(std::max<std::size_t>(efConstruction, 1)),
      levelMultiplier(1.0 / std::log(static_cast<double>(maxNeighbors))),
      rng(42),
      levelDistribution(0.0, 1.0)
{
}

int HnswIndex::RandomLevel()
{
    int level = 0;
    while (levelDistribution(rng) < levelMultiplier && level < 32)
    {
        ++level;
    }
    return level;
}

// 扁平邻接访问：每节点缓冲按层内联 [count, slots...]，层 0 容量 2*maxNeighbors、上层 maxNeighbors
std::size_t HnswIndex::LayerOffset(int layer) const
{
    return layer <= 0 ? 0 : 2 * maxNeighbors + 1 + static_cast<std::size_t>(layer - 1) * (maxNeighbors + 1);
}

std::size_t HnswIndex::LayerCapacity(int layer) const
{
    return layer == 0 ? 2 * maxNeighbors : maxNeighbors;
}

std::size_t HnswIndex::BufferLength(int level) const
{
    return 2 * maxNeighbors + 1 + static_cast<std::size_t>(level) * (maxNeighbors + 1);
}

HnswIndex::NeighborId* HnswIndex::LayerCountPtr(std::size_t node, int layer)
{
    return links[node].data() + LayerOffset(layer);
}

const HnswIndex::NeighborId* HnswIndex::LayerCountPtr(std::size_t node, int layer) const
{
    return links[node].data() + LayerOffset(layer);
}

HnswIndex::NeighborId* HnswIndex::LayerSlots(std::size_t node, int layer)
{
    return LayerCountPtr(node, layer) + 1;
}

const HnswIndex::NeighborId* HnswIndex::LayerSlots(std::size_t node, int layer) const
{
    return LayerCountPtr(node, layer) + 1;
}

std::size_t HnswIndex::LayerCount(std::size_t node, int layer) const
{
    return *LayerCountPtr(node, layer);
}

void HnswIndex::AddLink(int layer, std::size_t node, NeighborId neighbor)
{
    if (node >= nodeLevels.size() || layer < 0 || nodeLevels[node] < 0 ||
        static_cast<std::size_t>(layer) > static_cast<std::size_t>(nodeLevels[node]))
    {
        return;
    }
    const NeighborId nbr = neighbor;
    NeighborId* const count = LayerCountPtr(node, layer);
    NeighborId* const slots = LayerSlots(node, layer);
    for (std::size_t j = 0; j < *count; ++j)
    {
        if (slots[j] == nbr)
        {
            return;   // 已存在，幂等
        }
    }
    const std::size_t capacity = LayerCapacity(layer);
    if (*count < capacity)
    {
        slots[(*count)++] = nbr;
        return;
    }
    // 已满：替换"最远者"等价于"加入后保留最近 capacity 个"，免扩容
    const float* nvec = table->Vector(node);
    std::size_t weakest = 0;
    float weakest_score = DotProduct(nvec, table->Vector(slots[0]), dim);
    for (std::size_t j = 1; j < *count; ++j)
    {
        const float score = DotProduct(nvec, table->Vector(slots[j]), dim);
        if (score < weakest_score)
        {
            weakest_score = score;
            weakest = j;
        }
    }
    if (DotProduct(nvec, table->Vector(nbr), dim) > weakest_score)
    {
        slots[weakest] = nbr;
    }
}

// 单层 beam 搜索：从 entry 出发贪心扩展，返回与该层 query 最接近的至多 ef 个候选（降序）
std::vector<HnswIndex::Candidate> HnswIndex::SearchLayer(
    const float* query, std::size_t entry_id, std::size_t ef, int layer) const
{
    const auto score = [this, query](std::size_t id)
    {
        return DotProduct(query, table->Vector(id), dim);
    };
    // 候选集：大根堆按 score，堆顶 = 最近（优先探索）
    auto closestFirst = [](const Candidate& a, const Candidate& b)
    {
        return a.Score < b.Score;
    };
    // 结果集：小根堆，堆顶 = 最差（淘汰用）
    auto worstFirst = [](const Candidate& a, const Candidate& b)
    {
        return a.Score > b.Score;
    };

    // 已访问标记：函数级 thread_local buffer + 自增 generation，免每次清零 O(SlotCount())，
    // 且线程隔离 → 并发 Search 可共享读锁；跨实例共享安全（buffer 只增不减、标签按线程唯一）
    static thread_local std::vector<std::uint32_t> visitedTags;
    static thread_local std::uint32_t visitedGeneration = 0;
    if (visitedTags.size() < table->SlotCount())
    {
        visitedTags.assign(table->SlotCount(), 0);
    }
    const std::uint32_t tag = ++visitedGeneration;
    std::vector<std::uint32_t>& visited = visitedTags;
    visited[entry_id] = tag;

    // 堆内携带已算好的 score：只在发现节点时算一次，避免每次堆比较重算点积
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(closestFirst)>
        candidates(closestFirst);
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(worstFirst)>
        best(worstFirst);
    const Candidate entry{ entry_id, score(entry_id) };
    candidates.push(entry);
    best.push(entry);

    while (!candidates.empty())
    {
        const Candidate cur = candidates.top();   // 最近候选
        candidates.pop();
        if (best.size() >= ef && cur.Score < best.top().Score)
        {
            break;
        }
        const std::size_t layerCount = LayerCount(cur.Id, layer);
        const NeighborId* const layerSlots = LayerSlots(cur.Id, layer);
        for (std::size_t j = 0; j < layerCount; ++j)
        {
#if defined(__AVX2__)
            if (j + 1 < layerCount)
            {
                _mm_prefetch(reinterpret_cast<const char*>(table->Vector(layerSlots[j + 1])),
                             _MM_HINT_T0);
            }
#endif
            const std::size_t nbr = layerSlots[j];
            if (table->Deleted(nbr) || visited[nbr] == tag)
            {
                continue;
            }
            visited[nbr] = tag;
            const float nscore = score(nbr);
            if (best.size() < ef || nscore > best.top().Score)
            {
                candidates.push(Candidate{ nbr, nscore });
                best.push(Candidate{ nbr, nscore });
                if (best.size() > ef)
                {
                    best.pop();
                }
            }
        }
    }

    // best 堆顶最差；全部弹出为最差在前，反转得降序
    std::vector<Candidate> result;
    result.reserve(best.size());
    while (!best.empty())
    {
        result.push_back(best.top());
        best.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

// 启发式邻居选择：剔除与已选候选更近的冗余项，不足时用最近者补足
std::vector<std::size_t> HnswIndex::SelectNeighbors(
    const std::vector<Candidate>& candidates, std::size_t limit) const
{
    if (candidates.size() <= limit)
    {
        std::vector<std::size_t> all;
        all.reserve(candidates.size());
        for (const Candidate& c : candidates)
        {
            all.push_back(c.Id);
        }
        return all;
    }

    std::vector<std::size_t> result;
    result.reserve(limit);
    for (const Candidate& c : candidates)
    {
        if (result.size() >= limit)
        {
            break;
        }
        bool keep = true;
        for (std::size_t r : result)
        {
            const float dCr = DotProduct(table->Vector(c.Id), table->Vector(r), dim);
            if (dCr > c.Score)   // c 与 r 比与 query 更近 → c 冗余
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            result.push_back(c.Id);
        }
    }
    for (const Candidate& c : candidates)   // 启发式选不满则用最近补足
    {
        if (result.size() >= limit)
        {
            break;
        }
        if (std::find(result.begin(), result.end(), c.Id) == result.end())
        {
            result.push_back(c.Id);
        }
    }
    return result;
}

void HnswIndex::EraseFromLayer(std::size_t node, int layer, NeighborId neighbor)
{
    if (node >= nodeLevels.size() || nodeLevels[node] < 0)
    {
        return;
    }
    NeighborId* const count = LayerCountPtr(node, layer);
    NeighborId* const slots = LayerSlots(node, layer);
    for (std::size_t j = 0; j < *count; ++j)
    {
        if (slots[j] == neighbor)
        {
            std::move(slots + j + 1, slots + *count, slots + j);   // 左移保持顺序
            --*count;
            return;
        }
    }
}

void HnswIndex::Add(std::size_t id)
{
    if (table->Deleted(id))
    {
        return;
    }
    if (id < nodeLevels.size() && nodeLevels[id] >= 0)
    {
        Remove(id);   // 幂等：已在图中则先移除，再按当前向量重插
    }
    if (nodeLevels.size() <= id)
    {
        nodeLevels.resize(id + 1, -1);
        links.resize(id + 1);
    }

    const float* vec = table->Vector(id);
    const int newLevel = RandomLevel();
    nodeLevels[id] = newLevel;
    links[id].assign(BufferLength(newLevel), 0);

    if (enterPoint < 0)
    {
        enterPoint = static_cast<int>(id);
        topLevel = newLevel;
        return;
    }

    // 从最高层贪心下探到 newLevel 上方，收集入口
    int entry = enterPoint;
    for (int layer = topLevel; layer > newLevel; --layer)
    {
        const std::vector<Candidate> ep = SearchLayer(vec, static_cast<std::size_t>(entry), 1, layer);
        entry = static_cast<int>(ep.front().Id);
    }

    // 从 min(newLevel, topLevel) 到 0 逐层建立双向连接并修剪邻居
    for (int layer = std::min(newLevel, topLevel); layer >= 0; --layer)
    {
        const std::vector<Candidate> candidates = SearchLayer(vec, static_cast<std::size_t>(entry), efConstruction, layer);
        const std::vector<std::size_t> neighbors = SelectNeighbors(candidates, maxNeighbors);
        for (std::size_t nbr : neighbors)
        {
            AddLink(layer, id, nbr);
            AddLink(layer, nbr, id);
        }
        entry = static_cast<int>(candidates.front().Id);
    }

    if (newLevel > topLevel)
    {
        enterPoint = static_cast<int>(id);
        topLevel = newLevel;
    }
}

void HnswIndex::Remove(std::size_t id)
{
    if (id >= nodeLevels.size() || nodeLevels[id] < 0)
    {
        return;   // 不在图中
    }
    const int nodeLevel = nodeLevels[id];
    for (int layer = 0; layer <= nodeLevel; ++layer)
    {
        const std::size_t layerCount = LayerCount(id, layer);
        const NeighborId* const idSlots = LayerSlots(id, layer);
        std::vector<std::size_t> neighbors;
        neighbors.reserve(layerCount);
        for (std::size_t j = 0; j < layerCount; ++j)
        {
            const std::size_t nbr = idSlots[j];
            if (table->Deleted(nbr))
            {
                continue;
            }
            EraseFromLayer(nbr, layer, static_cast<NeighborId>(id));
            neighbors.push_back(nbr);
        }
        // 被删节点的邻居两两重连（双向）：每节点在"旧邻接 ∪ 其它邻居"中保留最近 capacity 个，
        // 与旧"先加链再按容量修剪"语义一致，防度无界膨胀
        const std::size_t capacity = LayerCapacity(layer);
        for (std::size_t i = 0; i < neighbors.size(); ++i)
        {
            std::vector<std::size_t> desired;
            desired.reserve(layerCount + neighbors.size());
            const std::size_t curCount = LayerCount(neighbors[i], layer);
            const NeighborId* const curSlots = LayerSlots(neighbors[i], layer);
            for (std::size_t k = 0; k < curCount; ++k)
            {
                desired.push_back(curSlots[k]);
            }
            for (std::size_t j = 0; j < neighbors.size(); ++j)
            {
                if (j != i &&
                    std::find(desired.begin(), desired.end(), neighbors[j]) == desired.end())
                {
                    desired.push_back(neighbors[j]);
                }
            }
            if (desired.size() > capacity)
            {
                const float* nvec = table->Vector(neighbors[i]);
                std::partial_sort(desired.begin(), desired.begin() + capacity, desired.end(),
                                  [this, nvec](std::size_t id_a, std::size_t id_b)
                                  {
                                      return DotProduct(nvec, table->Vector(id_a), dim) >
                                             DotProduct(nvec, table->Vector(id_b), dim);
                                  });
                desired.resize(capacity);
            }
            NeighborId* const count = LayerCountPtr(neighbors[i], layer);
            NeighborId* const slots = LayerSlots(neighbors[i], layer);
            *count = static_cast<NeighborId>(desired.size());
            for (std::size_t k = 0; k < desired.size(); ++k)
            {
                slots[k] = static_cast<NeighborId>(desired[k]);
            }
        }
    }
    nodeLevels[id] = -1;
    links[id].clear();
    if (enterPoint == static_cast<int>(id))
    {
        // 入口被删：退到任一层 0 存活节点
        enterPoint = -1;
        for (std::size_t i = 0; i < nodeLevels.size(); ++i)
        {
            if (nodeLevels[i] >= 0)
            {
                enterPoint = static_cast<int>(i);
                break;
            }
        }
        topLevel = enterPoint >= 0
            ? nodeLevels[static_cast<std::size_t>(enterPoint)]
            : 0;
    }
}

bool HnswIndex::Write(std::ostream& out) const
{
    const auto writeU64 = [&out](std::uint64_t value)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    const auto writeI32 = [&out](std::int32_t value)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };

    writeU64(static_cast<std::uint64_t>(maxNeighbors));
    writeU64(static_cast<std::uint64_t>(efConstruction));
    writeI32(static_cast<std::int32_t>(enterPoint));
    writeI32(static_cast<std::int32_t>(topLevel));
    writeU64(static_cast<std::uint64_t>(nodeLevels.size()));
    for (std::size_t id = 0; id < nodeLevels.size(); ++id)
    {
        writeI32(static_cast<std::int32_t>(nodeLevels[id]));
        if (nodeLevels[id] < 0)
        {
            continue;
        }
        for (std::size_t layer = 0;
             layer < static_cast<std::size_t>(nodeLevels[id] + 1); ++layer)
        {
            const std::size_t count = LayerCount(id, static_cast<int>(layer));
            const NeighborId* const slots = LayerSlots(id, static_cast<int>(layer));
            writeU64(static_cast<std::uint64_t>(count));
            for (std::size_t j = 0; j < count; ++j)
            {
                out.write(reinterpret_cast<const char*>(&slots[j]), sizeof(slots[j]));
            }
        }
    }
    return static_cast<bool>(out);
}

bool HnswIndex::Read(std::istream& in)
{
    const auto readU64 = [&in](std::uint64_t& value)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(in);
    };
    const auto readI32 = [&in](std::int32_t& value)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(in);
    };
    const auto readU32 = [&in](std::uint32_t& value)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(in);
    };

    std::uint64_t storedM = 0;
    std::uint64_t storedEfConstruction = 0;
    std::int32_t storedEnterPoint = 0;
    std::int32_t storedTopLevel = 0;
    std::uint64_t storedSlotCount = 0;
    if (!readU64(storedM) || !readU64(storedEfConstruction) || !readI32(storedEnterPoint) ||
        !readI32(storedTopLevel) || !readU64(storedSlotCount) ||
        storedSlotCount != table->SlotCount())
    {
        return false;
    }

    maxNeighbors = std::max<std::size_t>(storedM, 3);
    efConstruction = std::max<std::size_t>(storedEfConstruction, 1);
    levelMultiplier = 1.0 / std::log(static_cast<double>(maxNeighbors));

    nodeLevels.assign(static_cast<std::size_t>(storedSlotCount), -1);
    links.assign(static_cast<std::size_t>(storedSlotCount), {});
    for (std::size_t id = 0; id < nodeLevels.size(); ++id)
    {
        std::int32_t level = 0;
        if (!readI32(level) || level < -1 || level > 32)
        {
            return false;
        }
        if (level < 0)
        {
            continue;
        }
        if (table->Deleted(id))
        {
            return false;   // 图中不允许出现已删除槽位
        }
        nodeLevels[id] = static_cast<int>(level);
        links[id].assign(BufferLength(level), 0);
        for (std::size_t layer = 0; layer < static_cast<std::size_t>(level + 1); ++layer)
        {
            std::uint64_t neighborCount = 0;
            if (!readU64(neighborCount) || neighborCount > LayerCapacity(layer))
            {
                return false;
            }
            NeighborId* const count = LayerCountPtr(id, static_cast<int>(layer));
            NeighborId* const slots = LayerSlots(id, static_cast<int>(layer));
            *count = static_cast<NeighborId>(neighborCount);
            for (std::uint64_t i = 0; i < neighborCount; ++i)
            {
                NeighborId nbr = 0;
                if (!readU32(nbr) || static_cast<std::size_t>(nbr) >= nodeLevels.size())
                {
                    return false;   // 引用越界 → 判为无效
                }
                slots[i] = nbr;
            }
        }
    }

    if (storedEnterPoint < -1 || storedTopLevel < 0 ||
        (storedEnterPoint >= 0 &&
         (static_cast<std::size_t>(storedEnterPoint) >= nodeLevels.size() ||
          nodeLevels[static_cast<std::size_t>(storedEnterPoint)] < 0 ||
          storedTopLevel != nodeLevels[static_cast<std::size_t>(storedEnterPoint)])) ||
        NodeCount() != table->Count())
    {
        return false;   // 入口/层高不一致，或图中节点数 ≠ 存活向量数 → 判为无效
    }

    enterPoint = storedEnterPoint;
    topLevel = storedTopLevel;
    return true;
}

void HnswIndex::Rebuild()
{
    Clear();
    nodeLevels.assign(table->SlotCount(), -1);
    links.assign(table->SlotCount(), {});
    for (std::size_t id = 0; id < table->SlotCount(); ++id)
    {
        if (!table->Deleted(id))
        {
            Add(id);
        }
    }
}

void HnswIndex::Clear()
{
    nodeLevels.clear();
    links.clear();
    enterPoint = -1;
    topLevel = 0;
}

std::vector<Hit> HnswIndex::Search(const std::vector<float>& query, std::size_t k, std::size_t ef) const
{
    std::vector<Hit> result;
    if (enterPoint < 0 || k == 0 || query.size() != dim)
    {
        return result;
    }

    std::vector<float> q = query;
    if (metric == Metric::Cosine)
    {
        L2Normalize(q.data(), q.size());
    }

    // 从顶层贪心下探到第 0 层
    int entry = enterPoint;
    for (int layer = topLevel; layer > 0; --layer)
    {
        const std::vector<Candidate> ep = SearchLayer(q.data(), static_cast<std::size_t>(entry), 1, layer);
        entry = static_cast<int>(ep.front().Id);
    }

    const std::vector<Candidate> candidates =
        SearchLayer(q.data(), static_cast<std::size_t>(entry), std::max(ef, k), 0);
    const std::size_t cap = std::min(k, candidates.size());
    result.reserve(cap);
    for (std::size_t i = 0; i < cap; ++i)
    {
        result.push_back(Hit{ candidates[i].Id, candidates[i].Score });
    }
    return result;
}

std::size_t HnswIndex::NodeCount() const
{
    std::size_t n = 0;
    for (int level : nodeLevels)
    {
        if (level >= 0)
        {
            ++n;
        }
    }
    return n;
}

}  // namespace Beacon
