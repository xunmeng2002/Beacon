// HNSW 分层可导航小世界图索引：近似最近邻检索（独立层，不复制向量）
#pragma once
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <vector>

#include "Beacon/VectorTable.h"

namespace Beacon {

class HnswIndex
{
public:
    // 图引用 table 的向量与删除状态；维度/度量取自 table
    HnswIndex(const VectorTable* table, std::size_t m = 16, std::size_t efConstruction = 200);

    // 将 table 中一个存活向量增量插入图；若该 id 已在图中则先移除再按当前向量重插
    void Add(std::size_t id);

    // 将节点从图中移除（删除/重插共用）；不在图中则无操作
    void Remove(std::size_t id);

    // 清空并按 table 当前存活向量全量重建
    void Rebuild();

    // 近似 top-K；ef 为搜索宽度（越大越准越慢）
    std::vector<Hit> Search(const std::vector<float>& query, std::size_t k, std::size_t ef) const;

    // 序列化/反序列化图结构；Read 失败返回 false，调用方应标记重建
    bool Write(std::ostream& out) const;
    bool Read(std::istream& in);

    std::size_t NodeCount() const;
    void Clear();

private:
    using NeighborId = std::uint32_t;   // 链接表内存储的紧凑 id（4 字节，比 size_t 省一半）
    struct Candidate
    {
        std::size_t Id;
        float Score;
    };

    int RandomLevel();
    void AddLink(int layer, std::size_t node, NeighborId neighbor);
    void EraseFromLayer(std::size_t node, int layer, NeighborId neighbor);
    std::vector<Candidate> SearchLayer(const float* query, std::size_t entry_id, std::size_t ef, int layer) const;
    std::vector<std::size_t> SelectNeighbors(const std::vector<Candidate>& candidates, std::size_t limit) const;

    // 扁平邻接访问：每节点缓冲按层内联 [count, slots...]，层 0 容量 2*maxNeighbors、上层 maxNeighbors
    std::size_t LayerOffset(int layer) const;
    std::size_t LayerCapacity(int layer) const;
    std::size_t BufferLength(int level) const;
    NeighborId* LayerCountPtr(std::size_t node, int layer);
    const NeighborId* LayerCountPtr(std::size_t node, int layer) const;
    NeighborId* LayerSlots(std::size_t node, int layer);
    const NeighborId* LayerSlots(std::size_t node, int layer) const;
    std::size_t LayerCount(std::size_t node, int layer) const;

    const VectorTable* table;
    std::size_t dim;
    Metric metric;
    std::size_t maxNeighbors;
    std::size_t efConstruction;
    double levelMultiplier;                 // 1 / ln(maxNeighbors)

    std::vector<int> nodeLevels;            // 按 slot id，-1 = 未入图
    std::vector<std::vector<NeighborId>> links;   // [slot][扁平层缓冲：层 0 容量 2*maxNeighbors、上层 maxNeighbors，count 内联于层首]

    int enterPoint = -1;
    int topLevel = 0;

    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<double> levelDistribution;
};

}  // namespace Beacon
