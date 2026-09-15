#include "Beacon/HnswIndex.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

using namespace Beacon;

namespace
{
class HnswIndexTest : public ::testing::Test
{
protected:
    const std::size_t dim = 3;
    VectorTable table{ dim, Metric::Cosine };
    std::size_t xId = 0;
    std::size_t yId = 0;
    std::size_t zId = 0;

    void SetUp() override
    {
        xId = table.Add({ 1, 0, 0 }, "x");
        yId = table.Add({ 0, 1, 0 }, "y");
        zId = table.Add({ 0, 0, 1 }, "z");
    }
};
}  // namespace

TEST_F(HnswIndexTest, RebuildIndexesAllLiveVectors)
{
    HnswIndex index(&table, 8, 32);
    index.Rebuild();
    EXPECT_EQ(index.NodeCount(), 3u);

    // 删除一个向量后重建，节点数随存活数收缩
    table.Delete(yId);
    index.Rebuild();
    EXPECT_EQ(index.NodeCount(), 2u);
}

TEST_F(HnswIndexTest, SearchFindsAxisTopOneAndValidatesBounds)
{
    HnswIndex index(&table, 8, 32);
    index.Rebuild();

    const auto hits = index.Search({ 1, 0, 0 }, 3, 32);
    ASSERT_EQ(hits.size(), 3u);
    EXPECT_EQ(hits[0].Id, xId);                  // 查询 x 轴 → x 轴向量得分最高
    EXPECT_FLOAT_EQ(hits[0].Score, 1.0f);
    for (const Hit& h : hits)
    {
        EXPECT_LT(h.Id, table.SlotCount());      // id 全部在槽位范围内
    }

    // k 超出存活数返回全部；维度不匹配返回空
    EXPECT_EQ(index.Search({ 1, 0, 0 }, 10, 32).size(), 3u);
    EXPECT_TRUE(index.Search({ 1, 0 }, 3, 32).empty());
}

TEST_F(HnswIndexTest, RemoveDropsNodeAndIgnoresAbsentId)
{
    HnswIndex index(&table, 8, 32);
    index.Rebuild();
    EXPECT_EQ(index.NodeCount(), 3u);

    index.Remove(yId);
    EXPECT_EQ(index.NodeCount(), 2u);

    index.Remove(yId);   // 已移除 → 无操作
    index.Remove(99);      // 不在图中 → 无操作
    EXPECT_EQ(index.NodeCount(), 2u);

    const auto hits = index.Search({ 0, 1, 0 }, 1, 32);
    ASSERT_FALSE(hits.empty());
    EXPECT_NE(hits[0].Id, yId);   // 被移除节点不再返回
}

TEST_F(HnswIndexTest, WriteReadRoundTripPreservesGraph)
{
    HnswIndex index(&table, 8, 32);
    index.Rebuild();

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(index.Write(ss));

    HnswIndex restored(&table, 8, 32);   // 同一表 → SlotCount()/Count() 校验通过
    ASSERT_TRUE(restored.Read(ss));
    EXPECT_EQ(restored.NodeCount(), index.NodeCount());

    const auto before = index.Search({ 1, 0, 0 }, 3, 32);
    const auto after = restored.Search({ 1, 0, 0 }, 3, 32);
    ASSERT_EQ(after.size(), before.size());
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_EQ(after[i].Id, before[i].Id);
        EXPECT_FLOAT_EQ(after[i].Score, before[i].Score);
    }
}

TEST_F(HnswIndexTest, ReadRejectsTruncatedOrCorruptStream)
{
    HnswIndex index(&table, 8, 32);
    index.Rebuild();

    // 截断：仅保留 32 字节头部（8+8+4+4+8）→ 逐节点读取必然 EOF
    std::stringstream fullSs(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(index.Write(fullSs));
    const std::string full = fullSs.str();
    std::stringstream truncated(std::string(full.data(), 32),
                                std::ios::in | std::ios::out | std::ios::binary);
    HnswIndex truncatedRestored(&table, 8, 32);
    EXPECT_FALSE(truncatedRestored.Read(truncated));

    // 篡改：第一个节点的 level 字段（字节偏移 32）置为 >32 的非法值 → 拒绝
    std::stringstream corruptSs(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(index.Write(corruptSs));
    std::string bytes = corruptSs.str();
    bytes[32] = static_cast<char>(0x7F);
    bytes[33] = static_cast<char>(0xFF);
    bytes[34] = static_cast<char>(0xFF);
    bytes[35] = static_cast<char>(0xFF);
    std::stringstream corrupt(bytes, std::ios::in | std::ios::out | std::ios::binary);
    HnswIndex corruptRestored(&table, 8, 32);
    EXPECT_FALSE(corruptRestored.Read(corrupt));
}

TEST(HnswIndexEmptyGraph, SearchReturnsEmptyWithoutCrash)
{
    VectorTable table(3, Metric::Cosine);
    HnswIndex index(&table, 8, 32);   // 未 Rebuild → 空图
    EXPECT_EQ(index.NodeCount(), 0u);
    EXPECT_TRUE(index.Search({ 1, 0, 0 }, 5, 32).empty());
}
