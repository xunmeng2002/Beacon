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
    HnswIndex(const VectorTable* table, std::size_t m = 16, std::size_t efConstruction = 200);
    void Add(std::size_t id);
    void Remove(std::size_t id);
    void Rebuild();

    // 近似 top-K；ef 为搜索宽度（越大越准越慢）
    std::vector<Hit> Search(const std::vector<float>& query, std::size_t k, std::size_t ef) const;
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

    const VectorTable* table_;
    std::size_t dim_;
    Metric metric_;
    std::size_t maxNeighbors_;
    std::size_t efConstruction_;
    double levelMultiplier_;                 // 1 / ln(maxNeighbors)

    std::vector<int> nodeLevels_;            // 按 slot id，-1 = 未入图
    std::vector<std::vector<NeighborId>> links_;   // [slot][扁平层缓冲：层 0 容量 2*maxNeighbors、上层 maxNeighbors，count 内联于层首]

    int enterPoint_ = -1;
    int topLevel_ = 0;

    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> levelDistribution_;
};

}  // namespace Beacon
