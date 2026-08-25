#include "mdbvec/VectorDb.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
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

int main()
{
    DemoSmall();
    DemoLarge();
    DemoIndex();
    return 0;
}

// 3 维余弦检索 + CRUD + 持久化往返
static void DemoSmall()
{
    using namespace mdbvec;
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
    using namespace mdbvec;
    std::cout << "==== 演示 2：10k x 64 维暴力检索延迟 ====\n";

    const std::size_t count = 10000;
    const std::size_t dim = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::kCosine);
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
    using namespace mdbvec;
    std::cout << "==== 演示 3：HNSW 近似检索 vs 暴力检索 ====\n";

    const std::size_t count = 30000;
    const std::size_t dim = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    VectorDb db(dim, Metric::kCosine);
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < count; ++i)
    {
        for (auto& x : vec)
        {
            x = dist(rng);
        }
        db.Add(vec, "vec-" + std::to_string(i));
    }
    db.EnableIndex();
    std::cout << "  索引启用=" << db.IndexEnabled() << " 数据量=" << db.count() << "\n";

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
}
