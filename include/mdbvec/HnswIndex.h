// HNSW 分层可导航小世界图索引：近似最近邻检索（独立层，不复制向量）
#pragma once
#include <cstddef>
#include <iosfwd>
#include <random>
#include <vector>

#include "mdbvec/VectorTable.h"

namespace mdbvec {

class HnswIndex
{
public:
    // 图引用 table 的向量与删除状态；维度/度量取自 table
    HnswIndex(const VectorTable* table, std::size_t m = 16, std::size_t ef_construction = 200);

    // 将 table 中一个存活向量增量插入图；若该 id 已在图中则先移除再按当前向量重插
    void Add(std::size_t id);

    // 将节点从图中移除（删除/重插共用）；不在图中则无操作
    void Remove(std::size_t id);

    // 清空并按 table 当前存活向量全量重建
    void Rebuild();

    // 近似 top-K；ef 为搜索宽度（越大越准越慢）
    std::vector<Hit> Search(const std::vector<float>& query, std::size_t k,
                            std::size_t ef) const;

    // 序列化/反序列化图结构；Read 失败返回 false，调用方应标记重建
    bool Write(std::ostream& out) const;
    bool Read(std::istream& in);

    std::size_t node_count() const;
    void Clear();

private:
    struct Candidate
    {
        std::size_t id;
        float score;
    };

    int RandomLevel();
    void AddLink(int layer, std::size_t node, std::size_t neighbor);
    std::vector<Candidate> SearchLayer(const float* query, std::size_t entry_id, std::size_t ef, int layer) const;
    std::vector<std::size_t> SelectNeighbors(const std::vector<Candidate>& candidates, std::size_t limit) const;
    void PruneLinks(std::size_t node, int layer, std::size_t limit);

    const VectorTable* table_;
    std::size_t dim_;
    Metric metric_;
    std::size_t m_;
    std::size_t ef_construction_;
    double level_mult_;                     // 1 / ln(m)

    std::vector<int> node_level_;           // 按 slot id，-1 = 未入图
    std::vector<std::vector<std::vector<std::size_t>>> links_;   // [slot][layer][邻居]

    int enter_point_ = -1;
    int top_level_ = 0;

    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> level_rand_;
};

}  // namespace mdbvec
