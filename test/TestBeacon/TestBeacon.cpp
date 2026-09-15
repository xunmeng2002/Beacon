#include "Beacon/VectorDb.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

class Stopwatch
{
public:
    void Start()
    {
        begin = std::chrono::steady_clock::now();
    }

    double ElapsedMs() const
    {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now() - begin).count();
    }

private:
    std::chrono::steady_clock::time_point begin;
};

static void DemoSmall();
static void DemoLarge();
static void DemoIndex();
static void DemoConcurrent();

int main()
{
#ifdef NDEBUG
    const char* config = "Release";
#else
    const char* config = "Debug";
#endif
#ifdef __AVX2__
    const char* simd = "AVX2";
#else
    const char* simd = "标量回退";
#endif
    std::cout << "[构建配置] " << config << " / " << simd << "\n";

    DemoSmall();
    DemoLarge();
    DemoIndex();
    DemoConcurrent();
    return 0;
}

// 3 维余弦检索 + CRUD + 持久化往返
static void DemoSmall()
{
    using namespace Beacon;
    std::cout << "==== 演示 1：3 维余弦检索 + CRUD ====\n";

    const std::string path = "demo_small.beacon";

    // 向后兼容：加载已有旧文件（首跑为 v1 格式，验证 v1 -> v2 升级）
    VectorDb legacy(3, Metric::Cosine);
    if (legacy.Load(path))
    {
        const auto oldHits = legacy.SearchExact({ 0.95f, 0.05f, 0 }, 1);
        std::cout << "  旧文件加载成功 count=" << legacy.Count()
            << " top-1 meta=" << legacy.Metadata(oldHits[0].Id) << "\n";
    }

    VectorDb db(3, Metric::Cosine);
    db.Add({ 1, 0, 0 }, "x 轴");
    db.Add({ 0, 1, 0 }, "y 轴");
    db.Add({ 0, 0, 1 }, "z 轴");
    db.Add({ 0.9f, 0.1f, 0 }, "接近 x 轴");

    const auto hits = db.SearchExact({ 0.95f, 0.05f, 0 }, 2);
    for (const auto& h : hits)
    {
        std::cout << "  top id=" << h.Id << " score=" << h.Score
            << " meta=" << db.Metadata(h.Id) << "\n";
    }

    std::cout << "  --- 删除 id=3（接近 x 轴）---\n";
    db.Delete(3);
    const auto hits2 = db.SearchExact({ 0.95f, 0.05f, 0 }, 2);
    for (const auto& h : hits2)
    {
        std::cout << "  top id=" << h.Id << " meta=" << db.Metadata(h.Id) << "\n";
    }

    std::cout << "  --- 更新 id=2（z 轴）为接近 x 轴 ---\n";
    db.Update(2, { 0.9f, 0.1f, 0 }, "改造为接近 x 轴");
    const auto hits3 = db.SearchExact({ 0.95f, 0.05f, 0 }, 2);
    for (const auto& h : hits3)
    {
        std::cout << "  top id=" << h.Id << " meta=" << db.Metadata(h.Id) << "\n";
    }

    std::cout << "  --- 边界：越界删除 / 重复删除 / 维度不匹配 ---\n";
    std::cout << "  Delete(99)=" << db.Delete(99)
        << " Delete(3)=" << db.Delete(3)
        << " Update(2,{1,0})=" << db.Update(2, { 1, 0 }, "bad") << "\n";

    std::cout << "  存活数=" << db.Count() << " 持久化到 " << path << " ...\n";
    if (!db.Save(path))
    {
        std::cerr << "  保存失败\n";
        return;
    }
    VectorDb loaded(3, Metric::Cosine);
    if (!loaded.Load(path))
    {
        std::cerr << "  加载失败\n";
        return;
    }
    const auto hits4 = loaded.SearchExact({ 0.95f, 0.05f, 0 }, 2);
    std::cout << "  重载后 count=" << loaded.Count()
        << " deleted(3)=" << loaded.Deleted(3) << "\n";
    for (const auto& h : hits4)
    {
        std::cout << "  top id=" << h.Id << " meta=" << loaded.Metadata(h.Id) << "\n";
    }
}

// 10k 条 64 维暴力检索，量测延迟
static void DemoLarge()
{
    using namespace Beacon;
    std::cout << "==== 演示 2：10k x 64 维暴力检索延迟 ====\n";

    const std::size_t count = 10000;
    const std::size_t dim = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::Cosine);
    db.Reserve(count);   // 预分配连续存储，避免批量导入时重复扩容
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < count; ++i)
    {
        for (auto& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "vec-" + std::to_string(i));
    }

    for (auto& x : vec)
    {
        x = dist(rng);
    }
    Stopwatch sw;
    sw.Start();
    const auto hits = db.SearchExact(vec, 5);
    const double ms = sw.ElapsedMs();

    std::cout << "  数据量=" << db.Count() << " 维度=" << dim
        << " 检索耗时=" << ms << " ms\n";
    std::cout << "  top-1 id=" << hits[0].Id << " score=" << hits[0].Score << "\n";
}

// HNSW 近似检索 vs 暴力检索：recall@10 与延迟
static void DemoIndex()
{
    using namespace Beacon;
    std::cout << "==== 演示 3：HNSW 近似检索 vs 暴力检索 ====\n";

    const std::size_t count = 30000;
    const std::size_t dim = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::Cosine);
    db.Reserve(count);   // 预分配连续存储，避免批量导入时重复扩容
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < count; ++i)
    {
        for (auto& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "vec-" + std::to_string(i));
    }
    Stopwatch swBuild;
    swBuild.Start();
    db.EnableIndex();
    const double buildMs = swBuild.ElapsedMs();
    std::cout << "  索引启用=" << db.IndexEnabled() << " 数据量=" << db.Count()
              << " 构建耗时=" << buildMs << " ms\n";

    const std::size_t queries = 100;
    const std::size_t k = 10;
    const std::size_t ef = 100;

    double totalRecall = 0.0;
    std::mt19937 qrng(99);
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(qrng);
        }
        const auto exact = db.SearchExact(vec, k);
        const auto approx = db.SearchIndexed(vec, k, ef);
        std::size_t hit = 0;
        for (const auto& a : approx)
        {
            for (const auto& e : exact)
            {
                if (a.Id == e.Id)
                {
                    ++hit;
                    break;
                }
            }
        }
        totalRecall += static_cast<double>(hit) / k;
    }
    const double recall = totalRecall / queries;

    Stopwatch sw;
    sw.Start();
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(qrng);
        }
        db.SearchExact(vec, k);
    }
    const double bruteMs = sw.ElapsedMs() / queries;

    sw.Start();
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(qrng);
        }
        db.SearchIndexed(vec, k, ef);
    }
    const double indexMs = sw.ElapsedMs() / queries;

    std::cout << "  recall@10=" << recall
        << "  暴力=" << bruteMs << " ms/query"
        << "  HNSW=" << indexMs << " ms/query (ef=" << ef << ")\n";

    // 增量维护：删除 3000 个节点，量测耗时（对比懒重建整图 ~10s）与删除后 recall
    std::size_t removed = 0;
    Stopwatch swDel;
    swDel.Start();
    for (std::size_t id = 0; id < count && removed < 3000; ++id)
    {
        if (db.Delete(id))
        {
            ++removed;
        }
    }
    const double delMs = swDel.ElapsedMs();

    double totalRecallDel = 0.0;
    std::mt19937 drng(123);
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(drng);
        }
        const auto exact = db.SearchExact(vec, k);
        const auto approx = db.SearchIndexed(vec, k, ef);
        std::size_t hit = 0;
        for (const auto& a : approx)
        {
            for (const auto& e : exact)
            {
                if (a.Id == e.Id)
                {
                    ++hit;
                    break;
                }
            }
        }
        totalRecallDel += static_cast<double>(hit) / k;
    }

    // 含索引持久化并重载：首查若免重建（~0.2ms）即证明索引已从磁盘恢复
    const std::string ipath = "demo_index.beacon";
    db.Save(ipath);
    VectorDb reloaded(dim, Metric::Cosine);
    reloaded.Load(ipath);
    Stopwatch swFirst;
    swFirst.Start();
    const auto reloadHits = reloaded.SearchIndexed(vec, k, ef);
    const double firstMs = swFirst.ElapsedMs();

    std::cout << "  增量删 3000 节点耗时=" << delMs << " ms"
              << " 删除后 recall@10=" << (totalRecallDel / queries) << "\n";
    if (!reloadHits.empty())
    {
        std::cout << "  含索引重载后首查=" << firstMs << " ms (免重建) top1="
                  << reloadHits[0].Id << "\n";
    }
}

// 并发读写压力：4 读者并发 SearchIndexed + 2 写者 Add/Update/Delete。
// 断言最终 Count() == 初始 + 写者数×每写者Add − 写者数×每写者Delete；
// 锁保证每操作原子，故结果与线程交错顺序无关，可确定性校验。
static void DemoConcurrent()
{
    using namespace Beacon;
    std::cout << "==== 演示 4：并发读写（shared_mutex + thread_local visited）====\n";

    const std::size_t dim = 64;
    const std::size_t initial = 10000;
    std::mt19937 rng(2024);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::Cosine);
    db.Reserve(initial + 10000);
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < initial; ++i)
    {
        for (float& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "seed-" + std::to_string(i));
    }
    db.EnableIndex();

    const std::size_t numReaders = 4;
    const std::size_t numWriters = 2;
    const std::size_t addsPerWriter = 1000;
    const std::size_t deletesPerWriter = 500;
    const std::size_t expected =
        initial + numWriters * addsPerWriter - numWriters * deletesPerWriter;

    std::atomic<bool> stop{ false };
    std::vector<std::thread> threads;

    for (std::size_t r = 0; r < numReaders; ++r)
    {
        threads.emplace_back([&db, &stop, dim, seed = r]()
        {
            std::mt19937 qrng(1000 + seed);
            std::normal_distribution<float> qdist(0.0f, 1.0f);
            std::vector<float> q(dim);
            std::size_t queries = 0;
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
                ++queries;
            }
            std::cout << "  reader#" << seed << " 完成 " << queries << " 次检索\n";
        });
    }

    for (std::size_t w = 0; w < numWriters; ++w)
    {
        threads.emplace_back([&db, dim, seed = w, addsPerWriter, deletesPerWriter]()
        {
            std::mt19937 wrng(2000 + seed);
            std::normal_distribution<float> wdist(0.0f, 1.0f);
            std::vector<float> v(dim);
            for (std::size_t i = 0; i < addsPerWriter; ++i)
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
            for (std::size_t i = 0; i < deletesPerWriter; ++i)
            {
                db.Delete(seed * deletesPerWriter + i);   // 固定初值区段、各写者不相交、必成功
            }
        });
    }

    // 先 join 写者（界定总耗时），再通知读者停止并 join
    for (std::size_t w = 0; w < numWriters; ++w)
    {
        threads[w + numReaders].join();
    }
    stop.store(true, std::memory_order_relaxed);
    for (std::size_t r = 0; r < numReaders; ++r)
    {
        threads[r].join();
    }

    const std::size_t actual = db.Count();
    std::cout << "  最终 count=" << actual << " 期望=" << expected
              << "  " << (actual == expected ? "OK" : "FAIL") << "\n";
    if (actual != expected)
    {
        std::cerr << "  并发一致性断言失败！\n";
    }
}
