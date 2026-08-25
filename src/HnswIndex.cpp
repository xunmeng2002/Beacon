#include "mdbvec/HnswIndex.h"

#include "mdbvec/Metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

void HnswIndex::AddLink(int layer, std::size_t a, std::size_t b)
{
    if (layer < 0 || static_cast<std::size_t>(layer) >= links_[a].size())
    {
        return;
    }
    std::vector<std::size_t>& neighbors = links_[a][static_cast<std::size_t>(layer)];
    if (std::find(neighbors.begin(), neighbors.end(), b) == neighbors.end())
    {
        neighbors.push_back(b);
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
    std::sort(neighbors.begin(), neighbors.end(), [this, nvec](std::size_t a, std::size_t b)
    {
        return DotProduct(nvec, table_->vector(a), dim_) >
               DotProduct(nvec, table_->vector(b), dim_);
    });
    neighbors.resize(limit);
}

void HnswIndex::Add(std::size_t id)
{
    if (table_->deleted(id))
    {
        return;
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
        const std::vector<Candidate> ep =
            SearchLayer(vec, static_cast<std::size_t>(entry), 1, layer);
        entry = static_cast<int>(ep.front().id);
    }

    // 从 min(new_level, top_level_) 到 0 逐层建立双向连接并修剪邻居
    for (int layer = std::min(new_level, top_level_); layer >= 0; --layer)
    {
        const std::vector<Candidate> candidates =
            SearchLayer(vec, static_cast<std::size_t>(entry), ef_construction_, layer);
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

std::vector<Hit> HnswIndex::Search(const std::vector<float>& query, std::size_t k,
                                   std::size_t ef) const
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
        const std::vector<Candidate> ep =
            SearchLayer(q.data(), static_cast<std::size_t>(entry), 1, layer);
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
