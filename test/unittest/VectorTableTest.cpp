#include "Beacon/VectorTable.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

using namespace Beacon;

TEST(VectorTableTest, AddReturnsSequentialIdsAndRejectsDimMismatch)
{
    VectorTable table(3, Metric::InnerProduct);
    EXPECT_EQ(table.Add({ 1, 0, 0 }, "a"), 0u);
    EXPECT_EQ(table.Add({ 0, 1, 0 }, "b"), 1u);
    EXPECT_EQ(table.Add({ 0, 0, 1 }, "c"), 2u);
    EXPECT_EQ(table.Add({ 1, 0 }, "bad"), static_cast<std::size_t>(-1));
    EXPECT_EQ(table.Add({ 1, 0, 0, 0 }, "bad"), static_cast<std::size_t>(-1));
    EXPECT_EQ(table.Count(), 3u);
    EXPECT_EQ(table.SlotCount(), 3u);
}

TEST(VectorTableTest, CountAndSlotCountReflectLiveAndTotalSlots)
{
    VectorTable table(3, Metric::InnerProduct);
    table.Add({ 1, 0, 0 });
    table.Add({ 0, 1, 0 });
    table.Add({ 0, 0, 1 });
    table.Add({ 1, 1, 0 });
    EXPECT_EQ(table.Count(), 4u);
    EXPECT_EQ(table.SlotCount(), 4u);

    EXPECT_TRUE(table.Delete(1));
    EXPECT_EQ(table.Count(), 3u);        // 存活数 -1
    EXPECT_EQ(table.SlotCount(), 4u);   // 槽位不变（软删除）
}

TEST(VectorTableTest, DeleteTombstonesAndKeepsVectorReadable)
{
    VectorTable table(3, Metric::InnerProduct);
    table.Add({ 1, 0, 0 }, "x");
    table.Add({ 0, 1, 0 }, "y");

    EXPECT_TRUE(table.Delete(0));
    EXPECT_TRUE(table.Deleted(0));                    // tombstone 置位
    EXPECT_FALSE(table.Deleted(1));                   // 未删槽位不受影响
    EXPECT_FLOAT_EQ(table.Vector(0)[0], 1.0f);        // 数据不搬动，仍可读
    EXPECT_EQ(table.Metadata(0), "x");                // 元数据保留

    EXPECT_FALSE(table.Delete(0));                    // 重复删除
    EXPECT_FALSE(table.Delete(99));                   // 越界
    EXPECT_EQ(table.Count(), 1u);
}

TEST(VectorTableTest, UpdateOverwritesInPlaceAndRevivesTombstone)
{
    VectorTable table(3, Metric::InnerProduct);
    table.Add({ 1, 0, 0 }, "x");
    table.Add({ 0, 1, 0 }, "y");

    // 就地覆盖向量与元数据
    EXPECT_TRUE(table.Update(0, { 0, 0, 1 }, "z-ish"));
    EXPECT_FLOAT_EQ(table.Vector(0)[0], 0.0f);
    EXPECT_FLOAT_EQ(table.Vector(0)[2], 1.0f);
    EXPECT_EQ(table.Metadata(0), "z-ish");
    EXPECT_EQ(table.Count(), 2u);

    // 复活已删除槽位
    table.Delete(1);
    EXPECT_EQ(table.Count(), 1u);
    EXPECT_TRUE(table.Update(1, { 0, 0, 1 }, "revived"));
    EXPECT_FALSE(table.Deleted(1));
    EXPECT_EQ(table.Count(), 2u);
    EXPECT_EQ(table.Metadata(1), "revived");

    // 非法输入
    EXPECT_FALSE(table.Update(99, { 1, 0, 0 }));
    EXPECT_FALSE(table.Update(0, { 1, 0 }, "bad-dim"));
}

TEST(VectorTableTest, ReservePreallocatesWithoutChangingSemantics)
{
    VectorTable table(3, Metric::InnerProduct);
    table.Reserve(1024);
    EXPECT_EQ(table.SlotCount(), 0u);   // 预分配不产生槽位

    const std::size_t id = table.Add({ 1, 0, 0 }, "first");
    EXPECT_EQ(id, 0u);
    EXPECT_EQ(table.Count(), 1u);
    EXPECT_EQ(table.SlotCount(), 1u);
    EXPECT_FLOAT_EQ(table.Vector(id)[0], 1.0f);
}

TEST(VectorTableTest, SetDataRecomputesLiveCountFromDeletedFlags)
{
    VectorTable table(3, Metric::InnerProduct);
    std::vector<float> data = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };   // id 0/1 存活，id 2 已删
    table.SetData(3, Metric::InnerProduct, std::move(data),
                   { "x", "y", "z" }, { 0u, 0u, 1u });

    EXPECT_EQ(table.Dim(), 3u);
    EXPECT_EQ(table.Count(), 2u);        // 删除标志驱动存活数
    EXPECT_EQ(table.SlotCount(), 3u);
    EXPECT_TRUE(table.Deleted(2));
    EXPECT_FALSE(table.Deleted(0));
    EXPECT_EQ(table.Metadata(1), "y");
    EXPECT_FLOAT_EQ(table.Vector(2)[2], 1.0f);
}
