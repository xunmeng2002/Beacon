#include "MdbVector/HnswIndex.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

using namespace mdbvec;

namespace
{
class HnswIndexTest : public ::testing::Test
{
protected:
    const std::size_t dim_ = 3;
    VectorTable table_{ dim_, Metric::kCosine };
    std::size_t x_id_ = 0;
    std::size_t y_id_ = 0;
    std::size_t z_id_ = 0;

    void SetUp() override
    {
        x_id_ = table_.Add({ 1, 0, 0 }, "x");
        y_id_ = table_.Add({ 0, 1, 0 }, "y");
        z_id_ = table_.Add({ 0, 0, 1 }, "z");
    }
};
}  // namespace

TEST_F(HnswIndexTest, RebuildIndexesAllLiveVectors)
{
    HnswIndex index(&table_, 8, 32);
    index.Rebuild();
    EXPECT_EQ(index.node_count(), 3u);

    // 删除一个向量后重建，节点数随存活数收缩
    table_.Delete(y_id_);
    index.Rebuild();
    EXPECT_EQ(index.node_count(), 2u);
}

TEST_F(HnswIndexTest, SearchFindsAxisTopOneAndValidatesBounds)
{
    HnswIndex index(&table_, 8, 32);
    index.Rebuild();

    const auto hits = index.Search({ 1, 0, 0 }, 3, 32);
    ASSERT_EQ(hits.size(), 3u);
    EXPECT_EQ(hits[0].id, x_id_);                  // 查询 x 轴 → x 轴向量得分最高
    EXPECT_FLOAT_EQ(hits[0].score, 1.0f);
    for (const Hit& h : hits)
    {
        EXPECT_LT(h.id, table_.slot_count());      // id 全部在槽位范围内
    }

    // k 超出存活数返回全部；维度不匹配返回空
    EXPECT_EQ(index.Search({ 1, 0, 0 }, 10, 32).size(), 3u);
    EXPECT_TRUE(index.Search({ 1, 0 }, 3, 32).empty());
}

TEST_F(HnswIndexTest, RemoveDropsNodeAndIgnoresAbsentId)
{
    HnswIndex index(&table_, 8, 32);
    index.Rebuild();
    EXPECT_EQ(index.node_count(), 3u);

    index.Remove(y_id_);
    EXPECT_EQ(index.node_count(), 2u);

    index.Remove(y_id_);   // 已移除 → 无操作
    index.Remove(99);      // 不在图中 → 无操作
    EXPECT_EQ(index.node_count(), 2u);

    const auto hits = index.Search({ 0, 1, 0 }, 1, 32);
    ASSERT_FALSE(hits.empty());
    EXPECT_NE(hits[0].id, y_id_);   // 被移除节点不再返回
}

TEST_F(HnswIndexTest, WriteReadRoundTripPreservesGraph)
{
    HnswIndex index(&table_, 8, 32);
    index.Rebuild();

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(index.Write(ss));

    HnswIndex restored(&table_, 8, 32);   // 同一表 → slot_count/count 校验通过
    ASSERT_TRUE(restored.Read(ss));
    EXPECT_EQ(restored.node_count(), index.node_count());

    const auto before = index.Search({ 1, 0, 0 }, 3, 32);
    const auto after = restored.Search({ 1, 0, 0 }, 3, 32);
    ASSERT_EQ(after.size(), before.size());
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_EQ(after[i].id, before[i].id);
        EXPECT_FLOAT_EQ(after[i].score, before[i].score);
    }
}

TEST_F(HnswIndexTest, ReadRejectsTruncatedOrCorruptStream)
{
    HnswIndex index(&table_, 8, 32);
    index.Rebuild();

    // 截断：仅保留 32 字节头部（8+8+4+4+8）→ 逐节点读取必然 EOF
    std::stringstream full_ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(index.Write(full_ss));
    const std::string full = full_ss.str();
    std::stringstream truncated(std::string(full.data(), 32),
                                std::ios::in | std::ios::out | std::ios::binary);
    HnswIndex truncated_restored(&table_, 8, 32);
    EXPECT_FALSE(truncated_restored.Read(truncated));

    // 篡改：第一个节点的 level 字段（字节偏移 32）置为 >32 的非法值 → 拒绝
    std::stringstream corrupt_ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(index.Write(corrupt_ss));
    std::string bytes = corrupt_ss.str();
    bytes[32] = static_cast<char>(0x7F);
    bytes[33] = static_cast<char>(0xFF);
    bytes[34] = static_cast<char>(0xFF);
    bytes[35] = static_cast<char>(0xFF);
    std::stringstream corrupt(bytes, std::ios::in | std::ios::out | std::ios::binary);
    HnswIndex corrupt_restored(&table_, 8, 32);
    EXPECT_FALSE(corrupt_restored.Read(corrupt));
}

TEST(HnswIndexEmptyGraph, SearchReturnsEmptyWithoutCrash)
{
    VectorTable table(3, Metric::kCosine);
    HnswIndex index(&table, 8, 32);   // 未 Rebuild → 空图
    EXPECT_EQ(index.node_count(), 0u);
    EXPECT_TRUE(index.Search({ 1, 0, 0 }, 5, 32).empty());
}
