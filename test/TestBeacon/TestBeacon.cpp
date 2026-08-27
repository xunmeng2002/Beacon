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
        begin_ = std::chrono::steady_clock::now();
    }

    double elapsed_ms() const
    {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now() - begin_).count();
    }

private:
    std::chrono::steady_clock::time_point begin_;
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
    using namespace beacon;
    std::cout << "==== 演示 1：3 维余弦检索 + CRUD ====\n";

    const std::string path = "demo_small.mdbv";

    // 向后兼容：加载已有旧文件（首跑为 v1 格式，验证 v1 -> v2 升级）
    VectorDb legacy(3, Metric::kCosine);
    if (legacy.Load(path))
    {
        const auto old_hits = legacy.Search({ 0.95f, 0.05f, 0 }, 1);
        std::cout << "  旧文件加载成功 count=" << legacy.count()
            << " top-1 meta=" << legacy.metadata(old_hits[0].id) << "\n";
    }

    VectorDb db(3, Metric::kCosine);
    db.Add({ 1, 0, 0 }, "x 轴");
    db.Add({ 0, 1, 0 }, "y 轴");
    db.Add({ 0, 0, 1 }, "z 轴");
    db.Add({ 0.9f, 0.1f, 0 }, "接近 x 轴");

    const auto hits = db.Search({ 0.95f, 0.05f, 0 }, 2);
    for (const auto& h : hits)
    {
        std::cout << "  top id=" << h.id << " score=" << h.score
            << " meta=" << db.metadata(h.id) << "\n";
    }

    std::cout << "  --- 删除 id=3（接近 x 轴）---\n";
    db.Delete(3);
    const auto hits2 = db.Search({ 0.95f, 0.05f, 0 }, 2);
    for (const auto& h : hits2)
    {
        std::cout << "  top id=" << h.id << " meta=" << db.metadata(h.id) << "\n";
    }

    std::cout << "  --- 更新 id=2（z 轴）为接近 x 轴 ---\n";
    db.Update(2, { 0.9f, 0.1f, 0 }, "改造为接近 x 轴");
    const auto hits3 = db.Search({ 0.95f, 0.05f, 0 }, 2);
    for (const auto& h : hits3)
    {
        std::cout << "  top id=" << h.id << " meta=" << db.metadata(h.id) << "\n";
    }

    std::cout << "  --- 边界：越界删除 / 重复删除 / 维度不匹配 ---\n";
    std::cout << "  Delete(99)=" << db.Delete(99)
        << " Delete(3)=" << db.Delete(3)
        << " Update(2,{1,0})=" << db.Update(2, { 1, 0 }, "bad") << "\n";

    std::cout << "  存活数=" << db.count() << " 持久化到 " << path << " ...\n";
    if (!db.Save(path))
    {
        std::cerr << "  保存失败\n";
        return;
    }
    VectorDb loaded(3, Metric::kCosine);
    if (!loaded.Load(path))
    {
        std::cerr << "  加载失败\n";
        return;
    }
    const auto hits4 = loaded.Search({ 0.95f, 0.05f, 0 }, 2);
    std::cout << "  重载后 count=" << loaded.count()
        << " deleted(3)=" << loaded.deleted(3) << "\n";
    for (const auto& h : hits4)
    {
        std::cout << "  top id=" << h.id << " meta=" << loaded.metadata(h.id) << "\n";
    }
}

// 10k 条 64 维暴力检索，量测延迟
static void DemoLarge()
{
    using namespace beacon;
    std::cout << "==== 演示 2：10k x 64 维暴力检索延迟 ====\n";

    const std::size_t count = 10000;
    const std::size_t dim = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::kCosine);
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
    const auto hits = db.Search(vec, 5);
    const double ms = sw.elapsed_ms();

    std::cout << "  数据量=" << db.count() << " 维度=" << dim
        << " 检索耗时=" << ms << " ms\n";
    std::cout << "  top-1 id=" << hits[0].id << " score=" << hits[0].score << "\n";
}

// HNSW 近似检索 vs 暴力检索：recall@10 与延迟
static void DemoIndex()
{
    using namespace beacon;
    std::cout << "==== 演示 3：HNSW 近似检索 vs 暴力检索 ====\n";

    const std::size_t count = 30000;
    const std::size_t dim = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::kCosine);
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
    Stopwatch sw_build;
    sw_build.Start();
    db.EnableIndex();
    const double build_ms = sw_build.elapsed_ms();
    std::cout << "  索引启用=" << db.IndexEnabled() << " 数据量=" << db.count()
              << " 构建耗时=" << build_ms << " ms\n";

    const std::size_t queries = 100;
    const std::size_t k = 10;
    const std::size_t ef = 100;

    double total_recall = 0.0;
    std::mt19937 qrng(99);
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(qrng);
        }
        const auto exact = db.Search(vec, k);
        const auto approx = db.SearchIndexed(vec, k, ef);
        std::size_t hit = 0;
        for (const auto& a : approx)
        {
            for (const auto& e : exact)
            {
                if (a.id == e.id)
                {
                    ++hit;
                    break;
                }
            }
        }
        total_recall += static_cast<double>(hit) / k;
    }
    const double recall = total_recall / queries;

    Stopwatch sw;
    sw.Start();
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(qrng);
        }
        db.Search(vec, k);
    }
    const double brute_ms = sw.elapsed_ms() / queries;

    sw.Start();
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(qrng);
        }
        db.SearchIndexed(vec, k, ef);
    }
    const double index_ms = sw.elapsed_ms() / queries;

    std::cout << "  recall@10=" << recall
        << "  暴力=" << brute_ms << " ms/query"
        << "  HNSW=" << index_ms << " ms/query (ef=" << ef << ")\n";

    // 增量维护：删除 3000 个节点，量测耗时（对比懒重建整图 ~10s）与删除后 recall
    std::size_t removed = 0;
    Stopwatch sw_del;
    sw_del.Start();
    for (std::size_t id = 0; id < count && removed < 3000; ++id)
    {
        if (db.Delete(id))
        {
            ++removed;
        }
    }
    const double del_ms = sw_del.elapsed_ms();

    double total_recall_del = 0.0;
    std::mt19937 drng(123);
    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        for (auto& x : vec)
        {
            x = dist(drng);
        }
        const auto exact = db.Search(vec, k);
        const auto approx = db.SearchIndexed(vec, k, ef);
        std::size_t hit = 0;
        for (const auto& a : approx)
        {
            for (const auto& e : exact)
            {
                if (a.id == e.id)
                {
                    ++hit;
                    break;
                }
            }
        }
        total_recall_del += static_cast<double>(hit) / k;
    }

    // 含索引持久化并重载：首查若免重建（~0.2ms）即证明索引已从磁盘恢复
    const std::string ipath = "demo_index.mdbv";
    db.Save(ipath);
    VectorDb reloaded(dim, Metric::kCosine);
    reloaded.Load(ipath);
    Stopwatch sw_first;
    sw_first.Start();
    const auto reload_hits = reloaded.SearchIndexed(vec, k, ef);
    const double first_ms = sw_first.elapsed_ms();

    std::cout << "  增量删 3000 节点耗时=" << del_ms << " ms"
              << " 删除后 recall@10=" << (total_recall_del / queries) << "\n";
    if (!reload_hits.empty())
    {
        std::cout << "  含索引重载后首查=" << first_ms << " ms (免重建) top1="
                  << reload_hits[0].id << "\n";
    }
}

// 并发读写压力：4 读者并发 SearchIndexed + 2 写者 Add/Update/Delete。
// 断言最终 count == 初始 + 写者数×每写者Add − 写者数×每写者Delete；
// 锁保证每操作原子，故结果与线程交错顺序无关，可确定性校验。
static void DemoConcurrent()
{
    using namespace beacon;
    std::cout << "==== 演示 4：并发读写（shared_mutex + thread_local visited）====\n";

    const std::size_t dim = 64;
    const std::size_t initial = 10000;
    std::mt19937 rng(2024);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::kCosine);
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

    const std::size_t num_readers = 4;
    const std::size_t num_writers = 2;
    const std::size_t adds_per_writer = 1000;
    const std::size_t deletes_per_writer = 500;
    const std::size_t expected =
        initial + num_writers * adds_per_writer - num_writers * deletes_per_writer;

    std::atomic<bool> stop{ false };
    std::vector<std::thread> threads;

    for (std::size_t r = 0; r < num_readers; ++r)
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
                    db.metadata(hits[0].id);   // 按值返回，并发安全
                }
                ++queries;
            }
            std::cout << "  reader#" << seed << " 完成 " << queries << " 次检索\n";
        });
    }

    for (std::size_t w = 0; w < num_writers; ++w)
    {
        threads.emplace_back([&db, dim, seed = w, adds_per_writer, deletes_per_writer]()
        {
            std::mt19937 wrng(2000 + seed);
            std::normal_distribution<float> wdist(0.0f, 1.0f);
            std::vector<float> v(dim);
            for (std::size_t i = 0; i < adds_per_writer; ++i)
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
            for (std::size_t i = 0; i < deletes_per_writer; ++i)
            {
                db.Delete(seed * deletes_per_writer + i);   // 固定初值区段、各写者不相交、必成功
            }
        });
    }

    // 先 join 写者（界定总耗时），再通知读者停止并 join
    for (std::size_t w = 0; w < num_writers; ++w)
    {
        threads[w + num_readers].join();
    }
    stop.store(true, std::memory_order_relaxed);
    for (std::size_t r = 0; r < num_readers; ++r)
    {
        threads[r].join();
    }

    const std::size_t actual = db.count();
    std::cout << "  最终 count=" << actual << " 期望=" << expected
              << "  " << (actual == expected ? "OK" : "FAIL") << "\n";
    if (actual != expected)
    {
        std::cerr << "  并发一致性断言失败！\n";
    }
}
