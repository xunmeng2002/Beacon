#include "Beacon/VectorDb.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace Beacon;

namespace
{
constexpr char TestPath[] = "ut_db.beacon";

class VectorDbTest : public ::testing::Test
{
protected:
    const std::size_t dim = 3;
    VectorDb db{ dim, Metric::Cosine };
    std::size_t xId = 0;
    std::size_t yId = 0;
    std::size_t zId = 0;
    std::size_t nearXId = 0;

    void SetUp() override
    {
        xId = db.Add({ 1, 0, 0 }, "x 轴");
        yId = db.Add({ 0, 1, 0 }, "y 轴");
        zId = db.Add({ 0, 0, 1 }, "z 轴");
        nearXId = db.Add({ 0.9f, 0.1f, 0 }, "接近 x 轴");
    }

    void TearDown() override
    {
        std::remove(TestPath);
    }
};
}  // namespace

TEST_F(VectorDbTest, CrudSoftDeleteAndReviveKeepIdStable)
{
    EXPECT_EQ(db.Count(), 4u);
    EXPECT_EQ(db.Dim(), 3u);

    EXPECT_TRUE(db.Delete(yId));
    EXPECT_TRUE(db.Deleted(yId));
    EXPECT_FALSE(db.Deleted(xId));
    EXPECT_EQ(db.Count(), 3u);
    EXPECT_EQ(db.Metadata(yId), "y 轴");   // 数据不动，仅 tombstone

    EXPECT_FALSE(db.Delete(yId));          // 重复删除
    EXPECT_FALSE(db.Delete(99));             // 越界
    EXPECT_FALSE(db.Deleted(99));            // 越界视为不存在
    EXPECT_EQ(db.Metadata(99), "");          // 越界元数据返回空串

    EXPECT_TRUE(db.Update(yId, { 0, 0, 1 }, "复活"));
    EXPECT_FALSE(db.Deleted(yId));
    EXPECT_EQ(db.Count(), 4u);
    EXPECT_EQ(db.Metadata(yId), "复活");
}

TEST_F(VectorDbTest, SearchFindsAxisVectorExactly)
{
    const auto hits = db.SearchExact({ 1, 0, 0 }, 3);
    ASSERT_EQ(hits.size(), 3u);
    EXPECT_EQ(hits[0].Id, xId);
    EXPECT_FLOAT_EQ(hits[0].Score, 1.0f);
}

TEST_F(VectorDbTest, SearchKGreaterThanLiveCountReturnsAll)
{
    const auto hits = db.SearchExact({ 1, 0, 0 }, 100);
    EXPECT_EQ(hits.size(), db.Count());
}

TEST_F(VectorDbTest, SearchEmptyOrDimMismatchIsSafe)
{
    VectorDb empty(3, Metric::Cosine);
    EXPECT_TRUE(empty.SearchExact({ 1, 0, 0 }, 5).empty());
    EXPECT_TRUE(empty.SearchExact({ 1, 0, 0 }, 0).empty());

    EXPECT_TRUE(db.SearchExact({ 1, 0 }, 5).empty());          // 维度不匹配
    EXPECT_EQ(db.Add({ 1, 0 }, "bad"), static_cast<std::size_t>(-1));
    EXPECT_EQ(db.Update(nearXId, { 1, 0 }, "bad"), false);
}

TEST_F(VectorDbTest, EnableAndDisableIndexToggle)
{
    EXPECT_FALSE(db.IndexEnabled());
    db.EnableIndex();
    EXPECT_TRUE(db.IndexEnabled());
    EXPECT_FALSE(db.SearchIndexed({ 1, 0, 0 }, 1, 32).empty());

    db.DisableIndex();
    EXPECT_FALSE(db.IndexEnabled());
    EXPECT_TRUE(db.SearchIndexed({ 1, 0, 0 }, 1, 32).empty());   // 无索引 → 空
}

TEST_F(VectorDbTest, SaveLoadRoundTripPreservesDataAndIndex)
{
    db.EnableIndex();
    ASSERT_TRUE(db.Save(TestPath));

    VectorDb loaded(3, Metric::Cosine);
    ASSERT_TRUE(loaded.Load(TestPath));
    EXPECT_EQ(loaded.Count(), 4u);
    EXPECT_EQ(loaded.Dim(), 3u);
    EXPECT_EQ(loaded.Metadata(xId), "x 轴");
    EXPECT_EQ(loaded.Metadata(zId), "z 轴");
    EXPECT_FALSE(loaded.Deleted(yId));

    // 含索引保存 → 重载后索引从磁盘恢复，首查结果与暴力检索一致
    EXPECT_TRUE(loaded.IndexEnabled());
    const auto exact = db.SearchExact({ 1, 0, 0 }, 1);
    const auto approx = loaded.SearchIndexed({ 1, 0, 0 }, 1, 32);
    ASSERT_FALSE(approx.empty());
    EXPECT_EQ(approx[0].Id, exact[0].Id);
    EXPECT_FLOAT_EQ(approx[0].Score, exact[0].Score);
}

TEST_F(VectorDbTest, SaveLoadRoundTripPreservesTombstones)
{
    db.Delete(yId);
    ASSERT_TRUE(db.Save(TestPath));

    VectorDb loaded(3, Metric::Cosine);
    ASSERT_TRUE(loaded.Load(TestPath));
    EXPECT_EQ(loaded.Count(), 3u);
    EXPECT_TRUE(loaded.Deleted(yId));
    EXPECT_FALSE(loaded.Deleted(xId));
    EXPECT_EQ(loaded.Metadata(yId), "y 轴");
}

// 历史回归①：载入不同维度文件时，旧索引（维度不匹配）必须被丢弃重建
TEST(VectorDbRegression, LoadWithDimChangeDropsStaleIndex)
{
    const char* path = "ut_fourdim.beacon";
    const char* pathNo = "ut_fourdim_noindex.beacon";
    {
        // 4 维文件（含索引段）
        VectorDb small(4, Metric::Cosine);
        small.Add({ 1, 0, 0, 0 }, "x4");
        small.Add({ 0, 1, 0, 0 }, "y4");
        small.EnableIndex();
        ASSERT_TRUE(small.Save(path));

        // 无索引的 4 维文件
        VectorDb smallNo(4, Metric::Cosine);
        smallNo.Add({ 1, 0, 0, 0 }, "x4");
        ASSERT_TRUE(smallNo.Save(pathNo));
    }

    // 8 维库已启用索引，载入含索引段的 4 维文件 → 旧 8 维索引丢弃，恢复新 4 维索引
    VectorDb db8(8, Metric::Cosine);
    db8.Add({ 1, 0, 0, 0, 0, 0, 0, 0 }, "old8");
    db8.EnableIndex();
    ASSERT_TRUE(db8.Load(path));
    EXPECT_EQ(db8.Dim(), 4u);
    EXPECT_TRUE(db8.IndexEnabled());
    EXPECT_EQ(db8.Count(), 2u);
    const auto hits = db8.SearchIndexed({ 1, 0, 0, 0 }, 1, 32);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].Id, 0u);   // x4 轴

    // 无索引的 4 维文件 → 载入后既无旧索引也无新索引（旧索引已被丢弃）
    VectorDb db8b(8, Metric::Cosine);
    db8b.Add({ 1, 0, 0, 0, 0, 0, 0, 0 }, "old8");
    db8b.EnableIndex();
    ASSERT_TRUE(db8b.Load(pathNo));
    EXPECT_EQ(db8b.Dim(), 4u);
    EXPECT_FALSE(db8b.IndexEnabled());

    std::remove(path);
    std::remove(pathNo);
}

// 历史回归②：索引脏时 Save 不写索引段；Load 后首次 SearchIndexed 懒重建且结果正确
TEST(VectorDbRegression, DirtySaveOmitsIndexSegmentThenLazyRebuild)
{
    const char* pathNo = "ut_noindex.beacon";
    const char* pathDirty = "ut_dirty.beacon";
    {
        VectorDb seed(3, Metric::Cosine);
        seed.Add({ 1, 0, 0 }, "x");
        seed.Add({ 0, 1, 0 }, "y");
        seed.Add({ 0, 0, 1 }, "z");
        seed.Add({ 0.9f, 0.1f, 0 }, "near_x");
        ASSERT_TRUE(seed.Save(pathNo));   // 未启用索引 → 文件无索引段
    }
    {
        // 既有索引 + 载入无索引文件（维度匹配）→ indexDirty=true → Save 不写索引段
        VectorDb dirty(3, Metric::Cosine);
        dirty.EnableIndex();
        ASSERT_TRUE(dirty.Load(pathNo));
        ASSERT_TRUE(dirty.Save(pathDirty));
    }

    // 字节级守卫：脏存文件须与"未启用索引"基线等长——下面几条断言读侧只看 flag，
    // 索引段即便照写、尾部多余字节也会被静默忽略，拦不住写出条件被改回去
    const auto FileSize = [](const char* path)
    {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        return static_cast<std::streamoff>(in.tellg());
    };
    EXPECT_EQ(FileSize(pathDirty), FileSize(pathNo));

    // 载入到全新库：无索引段 → 索引未恢复（证明段被省略）
    VectorDb fresh(3, Metric::Cosine);
    ASSERT_TRUE(fresh.Load(pathDirty));
    EXPECT_FALSE(fresh.IndexEnabled());
    EXPECT_EQ(fresh.Count(), 4u);

    // 载入到已有索引的库：首次 SearchIndexed 触发懒重建，结果与暴力检索一致
    VectorDb reloaded(3, Metric::Cosine);
    reloaded.EnableIndex();
    ASSERT_TRUE(reloaded.Load(pathDirty));
    const auto exact = reloaded.SearchExact({ 1, 0, 0 }, 1);
    const auto approx = reloaded.SearchIndexed({ 1, 0, 0 }, 1, 32);
    ASSERT_FALSE(approx.empty());
    EXPECT_EQ(approx[0].Id, exact[0].Id);

    std::remove(pathNo);
    std::remove(pathDirty);
}

// 历史回归③：篡改索引段字节 → Load 仍成功（向量段权威），SearchIndexed 降级懒重建
TEST(VectorDbRegression, CorruptedIndexSegmentDegradesToLazyRebuild)
{
    const char* path = "ut_corrupt.beacon";
    {
        VectorDb src(3, Metric::Cosine);
        src.Add({ 1, 0, 0 }, "x");
        src.Add({ 0, 1, 0 }, "y");
        src.Add({ 0, 0, 1 }, "z");
        src.Add({ 0.9f, 0.1f, 0 }, "near_x");
        src.EnableIndex();
        ASSERT_TRUE(src.Save(path));
    }
    // 篡改文件尾部（索引段）：置 0xAB → 邻接 id 越界/计数非法，索引段校验必失败
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::size_t tail = bytes.size() - 32;
        for (std::size_t i = tail; i < bytes.size(); ++i)
        {
            bytes[i] = static_cast<char>(0xAB);
        }
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    VectorDb reloaded(3, Metric::Cosine);
    ASSERT_TRUE(reloaded.Load(path));          // 向量段权威 → Load 仍成功
    EXPECT_EQ(reloaded.Count(), 4u);
    const auto exact = reloaded.SearchExact({ 1, 0, 0 }, 1);
    const auto approx = reloaded.SearchIndexed({ 1, 0, 0 }, 1, 32);   // 校验失败 → 懒重建
    ASSERT_FALSE(approx.empty());
    EXPECT_EQ(approx[0].Id, exact[0].Id);

    std::remove(path);
}

// 近似检索召回率：固定种子 1000×32，recall@10 须 ≥ 0.8（宽松阈值防 flaky）
TEST(VectorDbIndexedSearch, RecallAgainstBruteForceExceedsThreshold)
{
    constexpr std::size_t Dim = 32;
    constexpr std::size_t Count = 1000;
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(Dim, Metric::Cosine);
    db.Reserve(Count);
    std::vector<float> vec(Dim);
    for (std::size_t i = 0; i < Count; ++i)
    {
        for (float& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "vec-" + std::to_string(i));
    }
    db.EnableIndex();

    constexpr std::size_t Queries = 50;
    constexpr std::size_t k = 10;
    std::size_t totalHits = 0;
    std::mt19937 qrng(999);
    for (std::size_t qi = 0; qi < Queries; ++qi)
    {
        for (float& x : vec)
        {
            x = dist(qrng);
        }
        const auto exact = db.SearchExact(vec, k);
        const auto approx = db.SearchIndexed(vec, k, 100);
        for (const Hit& a : approx)
        {
            for (const Hit& e : exact)
            {
                if (a.Id == e.Id)
                {
                    ++totalHits;
                    break;
                }
            }
        }
    }
    const double recall = static_cast<double>(totalHits) / (Queries * k);
    EXPECT_GE(recall, 0.8);
}

// 并发一致性：4 读者 + 2 写者；锁保证每操作原子，最终 Count() 与线程交错顺序无关
TEST(VectorDbConcurrent, ReadWriteConsistencyIsDeterministic)
{
    constexpr std::size_t Dim = 64;
    constexpr std::size_t Initial = 2000;
    std::mt19937 rng(2024);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(Dim, Metric::Cosine);
    db.Reserve(Initial + 5000);
    std::vector<float> vec(Dim);
    for (std::size_t i = 0; i < Initial; ++i)
    {
        for (float& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "seed-" + std::to_string(i));
    }
    db.EnableIndex();

    constexpr std::size_t Readers = 4;
    constexpr std::size_t Writers = 2;
    constexpr std::size_t AddsPerWriter = 500;
    constexpr std::size_t DeletesPerWriter = 200;
    const std::size_t expected =
        Initial + Writers * AddsPerWriter - Writers * DeletesPerWriter;

    std::atomic<bool> stop{ false };
    std::vector<std::thread> threads;
    for (std::size_t r = 0; r < Readers; ++r)
    {
        threads.emplace_back([&db, &stop, Dim, seed = r]()
        {
            std::mt19937 qrng(1000 + seed);
            std::normal_distribution<float> qdist(0.0f, 1.0f);
            std::vector<float> q(Dim);
            while (!stop.load(std::memory_order_relaxed))
            {
                for (float& x : q)
                {
                    x = qdist(qrng);
                }
                const std::vector<Hit> hits = db.SearchIndexed(q, 10, 64);
                if (!hits.empty())
                {
                    db.Metadata(hits[0].Id);   // 按值返回，并发安全
                }
            }
        });
    }
    for (std::size_t w = 0; w < Writers; ++w)
    {
        threads.emplace_back([&db, Dim, seed = w]()
        {
            std::mt19937 wrng(2000 + seed);
            std::normal_distribution<float> wdist(0.0f, 1.0f);
            std::vector<float> v(Dim);
            for (std::size_t i = 0; i < AddsPerWriter; ++i)
            {
                for (float& x : v)
                {
                    x = wdist(wrng);
                }
                const std::size_t id = db.Add(v, "w" + std::to_string(seed) + "-" + std::to_string(i));
                if (i % 5 == 0)
                {
                    for (float& x : v)
                    {
                        x = wdist(wrng);
                    }
                    db.Update(id, v, "upd");   // 覆盖本写者刚插入的活槽，Count() 不变
                }
            }
            for (std::size_t i = 0; i < DeletesPerWriter; ++i)
            {
                db.Delete(seed * DeletesPerWriter + i);   // 固定初值区段、各写者不相交、必成功
            }
        });
    }

    // 先 join 写者（界定总耗时），再通知读者停止并 join
    for (std::size_t w = 0; w < Writers; ++w)
    {
        threads[w + Readers].join();
    }
    stop.store(true, std::memory_order_relaxed);
    for (std::size_t r = 0; r < Readers; ++r)
    {
        threads[r].join();
    }

    EXPECT_EQ(db.Count(), expected);
}
