#include "Beacon/VectorTable.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

using namespace beacon;

TEST(VectorTableTest, AddReturnsSequentialIdsAndRejectsDimMismatch)
{
    VectorTable table(3, Metric::kInnerProduct);
    EXPECT_EQ(table.Add({ 1, 0, 0 }, "a"), 0u);
    EXPECT_EQ(table.Add({ 0, 1, 0 }, "b"), 1u);
    EXPECT_EQ(table.Add({ 0, 0, 1 }, "c"), 2u);
    EXPECT_EQ(table.Add({ 1, 0 }, "bad"), static_cast<std::size_t>(-1));
    EXPECT_EQ(table.Add({ 1, 0, 0, 0 }, "bad"), static_cast<std::size_t>(-1));
    EXPECT_EQ(table.count(), 3u);
    EXPECT_EQ(table.slot_count(), 3u);
}

TEST(VectorTableTest, CountAndSlotCountReflectLiveAndTotalSlots)
{
    VectorTable table(3, Metric::kInnerProduct);
    table.Add({ 1, 0, 0 });
    table.Add({ 0, 1, 0 });
    table.Add({ 0, 0, 1 });
    table.Add({ 1, 1, 0 });
    EXPECT_EQ(table.count(), 4u);
    EXPECT_EQ(table.slot_count(), 4u);

    EXPECT_TRUE(table.Delete(1));
    EXPECT_EQ(table.count(), 3u);        // 存活数 -1
    EXPECT_EQ(table.slot_count(), 4u);   // 槽位不变（软删除）
}

TEST(VectorTableTest, DeleteTombstonesAndKeepsVectorReadable)
{
    VectorTable table(3, Metric::kInnerProduct);
    table.Add({ 1, 0, 0 }, "x");
    table.Add({ 0, 1, 0 }, "y");

    EXPECT_TRUE(table.Delete(0));
    EXPECT_TRUE(table.deleted(0));                    // tombstone 置位
    EXPECT_FALSE(table.deleted(1));                   // 未删槽位不受影响
    EXPECT_FLOAT_EQ(table.vector(0)[0], 1.0f);        // 数据不搬动，仍可读
    EXPECT_EQ(table.metadata(0), "x");                // 元数据保留

    EXPECT_FALSE(table.Delete(0));                    // 重复删除
    EXPECT_FALSE(table.Delete(99));                   // 越界
    EXPECT_EQ(table.count(), 1u);
}

TEST(VectorTableTest, UpdateOverwritesInPlaceAndRevivesTombstone)
{
    VectorTable table(3, Metric::kInnerProduct);
    table.Add({ 1, 0, 0 }, "x");
    table.Add({ 0, 1, 0 }, "y");

    // 就地覆盖向量与元数据
    EXPECT_TRUE(table.Update(0, { 0, 0, 1 }, "z-ish"));
    EXPECT_FLOAT_EQ(table.vector(0)[0], 0.0f);
    EXPECT_FLOAT_EQ(table.vector(0)[2], 1.0f);
    EXPECT_EQ(table.metadata(0), "z-ish");
    EXPECT_EQ(table.count(), 2u);

    // 复活已删除槽位
    table.Delete(1);
    EXPECT_EQ(table.count(), 1u);
    EXPECT_TRUE(table.Update(1, { 0, 0, 1 }, "revived"));
    EXPECT_FALSE(table.deleted(1));
    EXPECT_EQ(table.count(), 2u);
    EXPECT_EQ(table.metadata(1), "revived");

    // 非法输入
    EXPECT_FALSE(table.Update(99, { 1, 0, 0 }));
    EXPECT_FALSE(table.Update(0, { 1, 0 }, "bad-dim"));
}

TEST(VectorTableTest, ReservePreallocatesWithoutChangingSemantics)
{
    VectorTable table(3, Metric::kInnerProduct);
    table.Reserve(1024);
    EXPECT_EQ(table.slot_count(), 0u);   // 预分配不产生槽位

    const std::size_t id = table.Add({ 1, 0, 0 }, "first");
    EXPECT_EQ(id, 0u);
    EXPECT_EQ(table.count(), 1u);
    EXPECT_EQ(table.slot_count(), 1u);
    EXPECT_FLOAT_EQ(table.vector(id)[0], 1.0f);
}

TEST(VectorTableTest, SetDataRecomputesLiveCountFromDeletedFlags)
{
    VectorTable table(3, Metric::kInnerProduct);
    std::vector<float> data = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };   // id 0/1 存活，id 2 已删
    table.set_data(3, Metric::kInnerProduct, std::move(data),
                   { "x", "y", "z" }, { 0u, 0u, 1u });

    EXPECT_EQ(table.dim(), 3u);
    EXPECT_EQ(table.count(), 2u);        // 删除标志驱动存活数
    EXPECT_EQ(table.slot_count(), 3u);
    EXPECT_TRUE(table.deleted(2));
    EXPECT_FALSE(table.deleted(0));
    EXPECT_EQ(table.metadata(1), "y");
    EXPECT_FLOAT_EQ(table.vector(2)[2], 1.0f);
}
