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

using namespace beacon;

namespace
{
constexpr char kTestPath[] = "ut_db.beacon";

class VectorDbTest : public ::testing::Test
{
protected:
    const std::size_t dim_ = 3;
    VectorDb db_{ dim_, Metric::kCosine };
    std::size_t x_id_ = 0;
    std::size_t y_id_ = 0;
    std::size_t z_id_ = 0;
    std::size_t near_x_id_ = 0;

    void SetUp() override
    {
        x_id_ = db_.Add({ 1, 0, 0 }, "x 轴");
        y_id_ = db_.Add({ 0, 1, 0 }, "y 轴");
        z_id_ = db_.Add({ 0, 0, 1 }, "z 轴");
        near_x_id_ = db_.Add({ 0.9f, 0.1f, 0 }, "接近 x 轴");
    }

    void TearDown() override
    {
        std::remove(kTestPath);
    }
};
}  // namespace

TEST_F(VectorDbTest, CrudSoftDeleteAndReviveKeepIdStable)
{
    EXPECT_EQ(db_.count(), 4u);
    EXPECT_EQ(db_.dim(), 3u);

    EXPECT_TRUE(db_.Delete(y_id_));
    EXPECT_TRUE(db_.deleted(y_id_));
    EXPECT_FALSE(db_.deleted(x_id_));
    EXPECT_EQ(db_.count(), 3u);
    EXPECT_EQ(db_.metadata(y_id_), "y 轴");   // 数据不动，仅 tombstone

    EXPECT_FALSE(db_.Delete(y_id_));          // 重复删除
    EXPECT_FALSE(db_.Delete(99));             // 越界
    EXPECT_FALSE(db_.deleted(99));            // 越界视为不存在
    EXPECT_EQ(db_.metadata(99), "");          // 越界元数据返回空串

    EXPECT_TRUE(db_.Update(y_id_, { 0, 0, 1 }, "复活"));
    EXPECT_FALSE(db_.deleted(y_id_));
    EXPECT_EQ(db_.count(), 4u);
    EXPECT_EQ(db_.metadata(y_id_), "复活");
}

TEST_F(VectorDbTest, SearchFindsAxisVectorExactly)
{
    const auto hits = db_.Search({ 1, 0, 0 }, 3);
    ASSERT_EQ(hits.size(), 3u);
    EXPECT_EQ(hits[0].id, x_id_);
    EXPECT_FLOAT_EQ(hits[0].score, 1.0f);
}

TEST_F(VectorDbTest, SearchKGreaterThanLiveCountReturnsAll)
{
    const auto hits = db_.Search({ 1, 0, 0 }, 100);
    EXPECT_EQ(hits.size(), db_.count());
}

TEST_F(VectorDbTest, SearchEmptyOrDimMismatchIsSafe)
{
    VectorDb empty(3, Metric::kCosine);
    EXPECT_TRUE(empty.Search({ 1, 0, 0 }, 5).empty());
    EXPECT_TRUE(empty.Search({ 1, 0, 0 }, 0).empty());

    EXPECT_TRUE(db_.Search({ 1, 0 }, 5).empty());          // 维度不匹配
    EXPECT_EQ(db_.Add({ 1, 0 }, "bad"), static_cast<std::size_t>(-1));
    EXPECT_EQ(db_.Update(near_x_id_, { 1, 0 }, "bad"), false);
}

TEST_F(VectorDbTest, EnableAndDisableIndexToggle)
{
    EXPECT_FALSE(db_.IndexEnabled());
    db_.EnableIndex();
    EXPECT_TRUE(db_.IndexEnabled());
    EXPECT_FALSE(db_.SearchIndexed({ 1, 0, 0 }, 1, 32).empty());

    db_.DisableIndex();
    EXPECT_FALSE(db_.IndexEnabled());
    EXPECT_TRUE(db_.SearchIndexed({ 1, 0, 0 }, 1, 32).empty());   // 无索引 → 空
}

TEST_F(VectorDbTest, SaveLoadRoundTripPreservesDataAndIndex)
{
    db_.EnableIndex();
    ASSERT_TRUE(db_.Save(kTestPath));

    VectorDb loaded(3, Metric::kCosine);
    ASSERT_TRUE(loaded.Load(kTestPath));
    EXPECT_EQ(loaded.count(), 4u);
    EXPECT_EQ(loaded.dim(), 3u);
    EXPECT_EQ(loaded.metadata(x_id_), "x 轴");
    EXPECT_EQ(loaded.metadata(z_id_), "z 轴");
    EXPECT_FALSE(loaded.deleted(y_id_));

    // 含索引保存 → 重载后索引从磁盘恢复，首查结果与暴力检索一致
    EXPECT_TRUE(loaded.IndexEnabled());
    const auto exact = db_.Search({ 1, 0, 0 }, 1);
    const auto approx = loaded.SearchIndexed({ 1, 0, 0 }, 1, 32);
    ASSERT_FALSE(approx.empty());
    EXPECT_EQ(approx[0].id, exact[0].id);
    EXPECT_FLOAT_EQ(approx[0].score, exact[0].score);
}

TEST_F(VectorDbTest, SaveLoadRoundTripPreservesTombstones)
{
    db_.Delete(y_id_);
    ASSERT_TRUE(db_.Save(kTestPath));

    VectorDb loaded(3, Metric::kCosine);
    ASSERT_TRUE(loaded.Load(kTestPath));
    EXPECT_EQ(loaded.count(), 3u);
    EXPECT_TRUE(loaded.deleted(y_id_));
    EXPECT_FALSE(loaded.deleted(x_id_));
    EXPECT_EQ(loaded.metadata(y_id_), "y 轴");
}

// 历史回归①：载入不同维度文件时，旧索引（维度不匹配）必须被丢弃重建
TEST(VectorDbRegression, LoadWithDimChangeDropsStaleIndex)
{
    const char* path = "ut_fourdim.beacon";
    const char* path_no = "ut_fourdim_noindex.beacon";
    {
        // 4 维文件（含索引段）
        VectorDb small(4, Metric::kCosine);
        small.Add({ 1, 0, 0, 0 }, "x4");
        small.Add({ 0, 1, 0, 0 }, "y4");
        small.EnableIndex();
        ASSERT_TRUE(small.Save(path));

        // 无索引的 4 维文件
        VectorDb small_no(4, Metric::kCosine);
        small_no.Add({ 1, 0, 0, 0 }, "x4");
        ASSERT_TRUE(small_no.Save(path_no));
    }

    // 8 维库已启用索引，载入含索引段的 4 维文件 → 旧 8 维索引丢弃，恢复新 4 维索引
    VectorDb db8(8, Metric::kCosine);
    db8.Add({ 1, 0, 0, 0, 0, 0, 0, 0 }, "old8");
    db8.EnableIndex();
    ASSERT_TRUE(db8.Load(path));
    EXPECT_EQ(db8.dim(), 4u);
    EXPECT_TRUE(db8.IndexEnabled());
    EXPECT_EQ(db8.count(), 2u);
    const auto hits = db8.SearchIndexed({ 1, 0, 0, 0 }, 1, 32);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].id, 0u);   // x4 轴

    // 无索引的 4 维文件 → 载入后既无旧索引也无新索引（旧索引已被丢弃）
    VectorDb db8b(8, Metric::kCosine);
    db8b.Add({ 1, 0, 0, 0, 0, 0, 0, 0 }, "old8");
    db8b.EnableIndex();
    ASSERT_TRUE(db8b.Load(path_no));
    EXPECT_EQ(db8b.dim(), 4u);
    EXPECT_FALSE(db8b.IndexEnabled());

    std::remove(path);
    std::remove(path_no);
}

// 历史回归②：索引脏时 Save 不写索引段；Load 后首次 SearchIndexed 懒重建且结果正确
TEST(VectorDbRegression, DirtySaveOmitsIndexSegmentThenLazyRebuild)
{
    const char* path_no = "ut_noindex.beacon";
    const char* path_dirty = "ut_dirty.beacon";
    {
        VectorDb seed(3, Metric::kCosine);
        seed.Add({ 1, 0, 0 }, "x");
        seed.Add({ 0, 1, 0 }, "y");
        seed.Add({ 0, 0, 1 }, "z");
        seed.Add({ 0.9f, 0.1f, 0 }, "near_x");
        ASSERT_TRUE(seed.Save(path_no));   // 未启用索引 → 文件无索引段
    }
    {
        // 既有索引 + 载入无索引文件（维度匹配）→ index_dirty_=true → Save 不写索引段
        VectorDb dirty(3, Metric::kCosine);
        dirty.EnableIndex();
        ASSERT_TRUE(dirty.Load(path_no));
        ASSERT_TRUE(dirty.Save(path_dirty));
    }

    // 载入到全新库：无索引段 → 索引未恢复（证明段被省略）
    VectorDb fresh(3, Metric::kCosine);
    ASSERT_TRUE(fresh.Load(path_dirty));
    EXPECT_FALSE(fresh.IndexEnabled());
    EXPECT_EQ(fresh.count(), 4u);

    // 载入到已有索引的库：首次 SearchIndexed 触发懒重建，结果与暴力检索一致
    VectorDb reloaded(3, Metric::kCosine);
    reloaded.EnableIndex();
    ASSERT_TRUE(reloaded.Load(path_dirty));
    const auto exact = reloaded.Search({ 1, 0, 0 }, 1);
    const auto approx = reloaded.SearchIndexed({ 1, 0, 0 }, 1, 32);
    ASSERT_FALSE(approx.empty());
    EXPECT_EQ(approx[0].id, exact[0].id);

    std::remove(path_no);
    std::remove(path_dirty);
}

// 历史回归③：篡改索引段字节 → Load 仍成功（向量段权威），SearchIndexed 降级懒重建
TEST(VectorDbRegression, CorruptedIndexSegmentDegradesToLazyRebuild)
{
    const char* path = "ut_corrupt.beacon";
    {
        VectorDb src(3, Metric::kCosine);
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

    VectorDb reloaded(3, Metric::kCosine);
    ASSERT_TRUE(reloaded.Load(path));          // 向量段权威 → Load 仍成功
    EXPECT_EQ(reloaded.count(), 4u);
    const auto exact = reloaded.Search({ 1, 0, 0 }, 1);
    const auto approx = reloaded.SearchIndexed({ 1, 0, 0 }, 1, 32);   // 校验失败 → 懒重建
    ASSERT_FALSE(approx.empty());
    EXPECT_EQ(approx[0].id, exact[0].id);

    std::remove(path);
}

// 近似检索召回率：固定种子 1000×32，recall@10 须 ≥ 0.8（宽松阈值防 flaky）
TEST(VectorDbIndexedSearch, RecallAgainstBruteForceExceedsThreshold)
{
    constexpr std::size_t kDim = 32;
    constexpr std::size_t kCount = 1000;
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(kDim, Metric::kCosine);
    db.Reserve(kCount);
    std::vector<float> vec(kDim);
    for (std::size_t i = 0; i < kCount; ++i)
    {
        for (float& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "vec-" + std::to_string(i));
    }
    db.EnableIndex();

    constexpr std::size_t kQueries = 50;
    constexpr std::size_t k = 10;
    std::size_t total_hits = 0;
    std::mt19937 qrng(999);
    for (std::size_t qi = 0; qi < kQueries; ++qi)
    {
        for (float& x : vec)
        {
            x = dist(qrng);
        }
        const auto exact = db.Search(vec, k);
        const auto approx = db.SearchIndexed(vec, k, 100);
        for (const Hit& a : approx)
        {
            for (const Hit& e : exact)
            {
                if (a.id == e.id)
                {
                    ++total_hits;
                    break;
                }
            }
        }
    }
    const double recall = static_cast<double>(total_hits) / (kQueries * k);
    EXPECT_GE(recall, 0.8);
}

// 并发一致性：4 读者 + 2 写者；锁保证每操作原子，最终 count 与线程交错顺序无关
TEST(VectorDbConcurrent, ReadWriteConsistencyIsDeterministic)
{
    constexpr std::size_t kDim = 64;
    constexpr std::size_t kInitial = 2000;
    std::mt19937 rng(2024);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(kDim, Metric::kCosine);
    db.Reserve(kInitial + 5000);
    std::vector<float> vec(kDim);
    for (std::size_t i = 0; i < kInitial; ++i)
    {
        for (float& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "seed-" + std::to_string(i));
    }
    db.EnableIndex();

    constexpr std::size_t kReaders = 4;
    constexpr std::size_t kWriters = 2;
    constexpr std::size_t kAddsPerWriter = 500;
    constexpr std::size_t kDeletesPerWriter = 200;
    const std::size_t expected =
        kInitial + kWriters * kAddsPerWriter - kWriters * kDeletesPerWriter;

    std::atomic<bool> stop{ false };
    std::vector<std::thread> threads;
    for (std::size_t r = 0; r < kReaders; ++r)
    {
        threads.emplace_back([&db, &stop, kDim, seed = r]()
        {
            std::mt19937 qrng(1000 + seed);
            std::normal_distribution<float> qdist(0.0f, 1.0f);
            std::vector<float> q(kDim);
            while (!stop.load(std::memory_order_relaxed))
            {
                for (float& x : q)
                {
                    x = qdist(qrng);
                }
                const std::vector<Hit> hits = db.SearchIndexed(q, 10, 64);
                if (!hits.empty())
                {
                    db.metadata(hits[0].id);   // 按值返回，并发安全
                }
            }
        });
    }
    for (std::size_t w = 0; w < kWriters; ++w)
    {
        threads.emplace_back([&db, kDim, seed = w]()
        {
            std::mt19937 wrng(2000 + seed);
            std::normal_distribution<float> wdist(0.0f, 1.0f);
            std::vector<float> v(kDim);
            for (std::size_t i = 0; i < kAddsPerWriter; ++i)
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
                    db.Update(id, v, "upd");   // 覆盖本写者刚插入的活槽，count 不变
                }
            }
            for (std::size_t i = 0; i < kDeletesPerWriter; ++i)
            {
                db.Delete(seed * kDeletesPerWriter + i);   // 固定初值区段、各写者不相交、必成功
            }
        });
    }

    // 先 join 写者（界定总耗时），再通知读者停止并 join
    for (std::size_t w = 0; w < kWriters; ++w)
    {
        threads[w + kReaders].join();
    }
    stop.store(true, std::memory_order_relaxed);
    for (std::size_t r = 0; r < kReaders; ++r)
    {
        threads[r].join();
    }

    EXPECT_EQ(db.count(), expected);
}
