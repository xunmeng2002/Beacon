#include "mdbvec/HnswIndex.h"

#include "mdbvec/Metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <queue>

namespace mdbvec {

HnswIndex::HnswIndex(const VectorTable* table, std::size_t m, std::size_t ef_construction)
    : table_(table),
      dim_(table->dim()),
      metric_(table->metric()),
      m_(std::max<std::size_t>(m, 3)),
      ef_construction_(std::max<std::size_t>(ef_construction, 1)),
      level_mult_(1.0 / std::log(static_cast<double>(m_))),
      rng_(42),
      level_rand_(0.0, 1.0)
{
}

int HnswIndex::RandomLevel()
{
    int level = 0;
    while (level_rand_(rng_) < level_mult_ && level < 32)
    {
        ++level;
    }
    return level;
}

void HnswIndex::AddLink(int layer, std::size_t node, std::size_t neighbor)
{
    if (layer < 0 || static_cast<std::size_t>(layer) >= links_[node].size())
    {
        return;
    }
    std::vector<std::size_t>& neighbors = links_[node][static_cast<std::size_t>(layer)];
    if (std::find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end())
    {
        neighbors.push_back(neighbor);
    }
}

// 单层 beam 搜索：从 entry 出发贪心扩展，返回与该层 query 最接近的至多 ef 个候选（降序）
std::vector<HnswIndex::Candidate> HnswIndex::SearchLayer(
    const float* query, std::size_t entry_id, std::size_t ef, int layer) const
{
    const auto score = [this, query](std::size_t id)
    {
        return DotProduct(query, table_->vector(id), dim_);
    };
    // 候选集：大根堆按 score，堆顶 = 最近（优先探索）
    auto closest_first = [](const Candidate& a, const Candidate& b)
    {
        return a.score < b.score;
    };
    // 结果集：小根堆，堆顶 = 最差（淘汰用）
    auto worst_first = [](const Candidate& a, const Candidate& b)
    {
        return a.score > b.score;
    };

    // 已访问标记（slot 数量小，位图代价低）
    std::vector<std::uint8_t> visited(table_->slot_count(), 0);
    visited[entry_id] = 1;

    // 堆内携带已算好的 score：只在发现节点时算一次，避免每次堆比较重算点积
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(closest_first)>
        candidates(closest_first);
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(worst_first)>
        best(worst_first);
    const Candidate entry{ entry_id, score(entry_id) };
    candidates.push(entry);
    best.push(entry);

    while (!candidates.empty())
    {
        const Candidate cur = candidates.top();   // 最近候选
        candidates.pop();
        if (best.size() >= ef && cur.score < best.top().score)
        {
            break;
        }
        for (std::size_t nbr : links_[cur.id][static_cast<std::size_t>(layer)])
        {
            if (table_->deleted(nbr) || visited[nbr])
            {
                continue;
            }
            visited[nbr] = 1;
            const float nscore = score(nbr);
            if (best.size() < ef || nscore > best.top().score)
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
            all.push_back(c.id);
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
            const float d_cr = DotProduct(table_->vector(c.id), table_->vector(r), dim_);
            if (d_cr > c.score)   // c 与 r 比与 query 更近 → c 冗余
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            result.push_back(c.id);
        }
    }
    for (const Candidate& c : candidates)   // 启发式选不满则用最近补足
    {
        if (result.size() >= limit)
        {
            break;
        }
        if (std::find(result.begin(), result.end(), c.id) == result.end())
        {
            result.push_back(c.id);
        }
    }
    return result;
}

void HnswIndex::PruneLinks(std::size_t node, int layer, std::size_t limit)
{
    std::vector<std::size_t>& neighbors = links_[node][static_cast<std::size_t>(layer)];
    if (neighbors.size() <= limit)
    {
        return;
    }
    const float* nvec = table_->vector(node);
    std::sort(neighbors.begin(), neighbors.end(), [this, nvec](std::size_t id_a, std::size_t id_b)
    {
        return DotProduct(nvec, table_->vector(id_a), dim_) >
               DotProduct(nvec, table_->vector(id_b), dim_);
    });
    neighbors.resize(limit);
}

void HnswIndex::Add(std::size_t id)
{
    if (table_->deleted(id))
    {
        return;
    }
    if (id < node_level_.size() && node_level_[id] >= 0)
    {
        Remove(id);   // 幂等：已在图中则先移除，再按当前向量重插
    }
    if (node_level_.size() <= id)
    {
        node_level_.resize(id + 1, -1);
        links_.resize(id + 1);
    }

    const float* vec = table_->vector(id);
    const int new_level = RandomLevel();
    node_level_[id] = new_level;
    links_[id].assign(static_cast<std::size_t>(new_level) + 1, {});

    if (enter_point_ < 0)
    {
        enter_point_ = static_cast<int>(id);
        top_level_ = new_level;
        return;
    }

    // 从最高层贪心下探到 new_level 上方，收集入口
    int entry = enter_point_;
    for (int layer = top_level_; layer > new_level; --layer)
    {
        const std::vector<Candidate> ep = SearchLayer(vec, static_cast<std::size_t>(entry), 1, layer);
        entry = static_cast<int>(ep.front().id);
    }

    // 从 min(new_level, top_level_) 到 0 逐层建立双向连接并修剪邻居
    for (int layer = std::min(new_level, top_level_); layer >= 0; --layer)
    {
        const std::vector<Candidate> candidates = SearchLayer(vec, static_cast<std::size_t>(entry), ef_construction_, layer);
        const std::vector<std::size_t> neighbors = SelectNeighbors(candidates, m_);
        for (std::size_t nbr : neighbors)
        {
            AddLink(layer, id, nbr);
            AddLink(layer, nbr, id);
        }
        const std::size_t max_links = (layer == 0) ? 2 * m_ : m_;
        for (std::size_t nbr : neighbors)
        {
            PruneLinks(nbr, layer, max_links);
        }
        entry = static_cast<int>(candidates.front().id);
    }

    if (new_level > top_level_)
    {
        enter_point_ = static_cast<int>(id);
        top_level_ = new_level;
    }
}

void HnswIndex::Remove(std::size_t id)
{
    if (id >= node_level_.size() || node_level_[id] < 0)
    {
        return;   // 不在图中
    }
    const int node_level = node_level_[id];
    for (int layer = 0; layer <= node_level; ++layer)
    {
        const auto& id_links = links_[id][static_cast<std::size_t>(layer)];
        std::vector<std::size_t> neighbors;
        neighbors.reserve(id_links.size());
        for (std::size_t nbr : id_links)
        {
            if (table_->deleted(nbr))
            {
                continue;
            }
            // 从 nbr 的该层邻接表剔除 id
            std::vector<std::size_t>& nlist = links_[nbr][static_cast<std::size_t>(layer)];
            nlist.erase(std::remove(nlist.begin(), nlist.end(), id), nlist.end());
            neighbors.push_back(nbr);
        }
        // 被删节点的邻居两两重连（双向），保持该层连通性；随后按容量修剪，防度无界膨胀
        const std::size_t max_links = (layer == 0) ? 2 * m_ : m_;
        for (std::size_t i = 0; i < neighbors.size(); ++i)
        {
            for (std::size_t j = i + 1; j < neighbors.size(); ++j)
            {
                AddLink(layer, neighbors[i], neighbors[j]);
                AddLink(layer, neighbors[j], neighbors[i]);
            }
        }
        for (std::size_t nbr : neighbors)
        {
            PruneLinks(nbr, layer, max_links);
        }
    }
    node_level_[id] = -1;
    links_[id].clear();
    if (enter_point_ == static_cast<int>(id))
    {
        // 入口被删：退到任一层 0 存活节点
        enter_point_ = -1;
        for (std::size_t i = 0; i < node_level_.size(); ++i)
        {
            if (node_level_[i] >= 0)
            {
                enter_point_ = static_cast<int>(i);
                break;
            }
        }
        top_level_ = enter_point_ >= 0
            ? node_level_[static_cast<std::size_t>(enter_point_)]
            : 0;
    }
}

bool HnswIndex::Write(std::ostream& out) const
{
    const auto write_u64 = [&out](std::uint64_t value)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    const auto write_i32 = [&out](std::int32_t value)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };

    write_u64(static_cast<std::uint64_t>(m_));
    write_u64(static_cast<std::uint64_t>(ef_construction_));
    write_i32(static_cast<std::int32_t>(enter_point_));
    write_i32(static_cast<std::int32_t>(top_level_));
    write_u64(static_cast<std::uint64_t>(node_level_.size()));
    for (std::size_t id = 0; id < node_level_.size(); ++id)
    {
        write_i32(static_cast<std::int32_t>(node_level_[id]));
        if (node_level_[id] < 0)
        {
            continue;
        }
        for (std::size_t layer = 0;
             layer < static_cast<std::size_t>(node_level_[id] + 1); ++layer)
        {
            const auto& neighbors = links_[id][layer];
            write_u64(static_cast<std::uint64_t>(neighbors.size()));
            for (std::size_t nbr : neighbors)
            {
                const std::uint32_t value = static_cast<std::uint32_t>(nbr);
                out.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
    }
    return static_cast<bool>(out);
}

bool HnswIndex::Read(std::istream& in)
{
    const auto read_u64 = [&in](std::uint64_t& value)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(in);
    };
    const auto read_i32 = [&in](std::int32_t& value)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(in);
    };
    const auto read_u32 = [&in](std::uint32_t& value)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(in);
    };

    std::uint64_t m = 0;
    std::uint64_t ef_construction = 0;
    std::int32_t enter_point = 0;
    std::int32_t top_level = 0;
    std::uint64_t slot_count = 0;
    if (!read_u64(m) || !read_u64(ef_construction) || !read_i32(enter_point) ||
        !read_i32(top_level) || !read_u64(slot_count) ||
        slot_count != table_->slot_count())
    {
        return false;
    }

    m_ = std::max<std::size_t>(m, 3);
    ef_construction_ = std::max<std::size_t>(ef_construction, 1);
    level_mult_ = 1.0 / std::log(static_cast<double>(m_));

    node_level_.assign(static_cast<std::size_t>(slot_count), -1);
    links_.assign(static_cast<std::size_t>(slot_count), {});
    for (std::size_t id = 0; id < node_level_.size(); ++id)
    {
        std::int32_t level = 0;
        if (!read_i32(level) || level < -1 || level > 32)
        {
            return false;
        }
        if (level < 0)
        {
            continue;
        }
        if (table_->deleted(id))
        {
            return false;   // 图中不允许出现已删除槽位
        }
        node_level_[id] = static_cast<int>(level);
        links_[id].assign(static_cast<std::size_t>(level) + 1, {});
        for (std::size_t layer = 0; layer < static_cast<std::size_t>(level + 1); ++layer)
        {
            std::uint64_t neighbor_count = 0;
            if (!read_u64(neighbor_count) || neighbor_count > 2 * m_)
            {
                return false;
            }
            std::vector<std::size_t>& nlist = links_[id][layer];
            nlist.reserve(static_cast<std::size_t>(neighbor_count));
            for (std::uint64_t i = 0; i < neighbor_count; ++i)
            {
                std::uint32_t nbr = 0;
                if (!read_u32(nbr) || static_cast<std::size_t>(nbr) >= node_level_.size())
                {
                    return false;   // 引用越界 → 判为无效
                }
                nlist.push_back(static_cast<std::size_t>(nbr));
            }
        }
    }

    if (enter_point < -1 || top_level < 0 ||
        (enter_point >= 0 &&
         (static_cast<std::size_t>(enter_point) >= node_level_.size() ||
          node_level_[static_cast<std::size_t>(enter_point)] < 0 ||
          top_level != node_level_[static_cast<std::size_t>(enter_point)])) ||
        node_count() != table_->count())
    {
        return false;   // 入口/层高不一致，或图中节点数 ≠ 存活向量数 → 判为无效
    }

    enter_point_ = enter_point;
    top_level_ = top_level;
    return true;
}

void HnswIndex::Rebuild()
{
    Clear();
    node_level_.assign(table_->slot_count(), -1);
    links_.assign(table_->slot_count(), {});
    for (std::size_t id = 0; id < table_->slot_count(); ++id)
    {
        if (!table_->deleted(id))
        {
            Add(id);
        }
    }
}

void HnswIndex::Clear()
{
    node_level_.clear();
    links_.clear();
    enter_point_ = -1;
    top_level_ = 0;
}

std::vector<Hit> HnswIndex::Search(const std::vector<float>& query, std::size_t k, std::size_t ef) const
{
    std::vector<Hit> result;
    if (enter_point_ < 0 || k == 0 || query.size() != dim_)
    {
        return result;
    }

    std::vector<float> q = query;
    if (metric_ == Metric::kCosine)
    {
        L2Normalize(q.data(), q.size());
    }

    // 从顶层贪心下探到第 0 层
    int entry = enter_point_;
    for (int layer = top_level_; layer > 0; --layer)
    {
        const std::vector<Candidate> ep = SearchLayer(q.data(), static_cast<std::size_t>(entry), 1, layer);
        entry = static_cast<int>(ep.front().id);
    }

    const std::vector<Candidate> candidates =
        SearchLayer(q.data(), static_cast<std::size_t>(entry), std::max(ef, k), 0);
    const std::size_t cap = std::min(k, candidates.size());
    result.reserve(cap);
    for (std::size_t i = 0; i < cap; ++i)
    {
        result.push_back(Hit{ candidates[i].id, candidates[i].score });
    }
    return result;
}

std::size_t HnswIndex::node_count() const
{
    std::size_t n = 0;
    for (int level : node_level_)
    {
        if (level >= 0)
        {
            ++n;
        }
    }
    return n;
}

}  // namespace mdbvec
