// Beacon（自研 HNSW）与 hnswlib 基准对比：同一数据/查询下比 build_ms / recall@k / ms-query
// 构建需 vcpkg toolchain（BEACON_ENABLE_BENCH=ON），见 CMakeLists 与 README。
#include "Beacon/Metrics.h"
#include "Beacon/VectorDb.h"

#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

double NowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct BenchParams
{
    std::size_t Count;
    std::size_t Dim;
    std::size_t MaxNeighbors;
    std::size_t EfConstruction;
    std::size_t Ef;
    std::size_t K;
    std::size_t QueryCount;
};

// 数据 = 原样（喂 Beacon，内部余弦归一化）+ L2 归一化副本（喂 hnswlib 与暴力基准）
struct Dataset
{
    std::size_t Count;
    std::size_t Dim;
    std::vector<float> Raw;
    std::vector<float> Normalized;
};

Dataset GenerateDataset(std::size_t count, std::size_t dim, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(count * dim);
    std::vector<float> normalized(count * dim);
    std::vector<float> row(dim);
    for (std::size_t id = 0; id < count; ++id)
    {
        for (float& x : row)
        {
            x = dist(rng);
        }
        std::copy(row.begin(), row.end(), raw.begin() + static_cast<std::ptrdiff_t>(id * dim));
        Beacon::L2Normalize(row.data(), dim);
        std::copy(row.begin(), row.end(), normalized.begin() + static_cast<std::ptrdiff_t>(id * dim));
    }
    return Dataset{ count, dim, std::move(raw), std::move(normalized) };
}

std::vector<std::vector<float>> GenerateQueries(std::size_t count, std::size_t dim, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> queries;
    queries.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        std::vector<float> q(dim);
        for (float& x : q)
        {
            x = dist(rng);
        }
        queries.push_back(std::move(q));
    }
    return queries;
}

// 独立暴力 top-K（归一化数据点积 = 余弦），作 ground truth；返回命中 id 集合（顺序无关）
std::vector<std::size_t> BruteForceTopK(const std::vector<float>& queryNorm, const Dataset& ds,
                                        std::size_t k)
{
    struct WorstFirst
    {
        bool operator()(const std::pair<float, std::size_t>& a,
                        const std::pair<float, std::size_t>& b) const
        {
            return a.first > b.first;   // 堆顶 = 最差候选
        }
    };
    std::priority_queue<std::pair<float, std::size_t>,
                        std::vector<std::pair<float, std::size_t>>, WorstFirst> heap;
    for (std::size_t id = 0; id < ds.Count; ++id)
    {
        const float score = Beacon::DotProduct(queryNorm.data(),
                                               ds.Normalized.data() + id * ds.Dim, ds.Dim);
        if (heap.size() < k)
        {
            heap.push(std::make_pair(score, id));
        }
        else if (score > heap.top().first)
        {
            heap.pop();
            heap.push(std::make_pair(score, id));
        }
    }
    std::vector<std::size_t> ids;
    ids.reserve(heap.size());
    while (!heap.empty())
    {
        ids.push_back(heap.top().second);
        heap.pop();
    }
    return ids;
}

std::size_t OverlapCount(const std::vector<std::size_t>& approx, const std::vector<std::size_t>& exact)
{
    std::size_t hit = 0;
    for (std::size_t id : approx)
    {
        if (std::find(exact.begin(), exact.end(), id) != exact.end())
        {
            ++hit;
        }
    }
    return hit;
}

struct BenchResult
{
    double BuildMs;
    double QueryMsTotal;
    double Recall;
};

BenchResult RunBeacon(const Dataset& ds, const std::vector<std::vector<float>>& rawQueries,
                         const BenchParams& p, const std::vector<std::vector<std::size_t>>& groundTruth)
{
    using namespace Beacon;
    VectorDb db(p.Dim, Metric::Cosine);
    db.Reserve(p.Count);
    std::vector<float> row(p.Dim);
    for (std::size_t id = 0; id < p.Count; ++id)
    {
        std::copy(ds.Raw.data() + id * p.Dim, ds.Raw.data() + (id + 1) * p.Dim, row.begin());
        db.Add(row, "");
    }

    const double tBuild = NowMs();
    db.EnableIndex(p.MaxNeighbors, p.EfConstruction);
    const double buildMs = NowMs() - tBuild;

    std::size_t totalHit = 0;
    const double tQuery = NowMs();
    for (std::size_t i = 0; i < p.QueryCount; ++i)
    {
        const std::vector<Hit> hits = db.SearchIndexed(rawQueries[i], p.K, p.Ef);
        std::vector<std::size_t> ids;
        ids.reserve(hits.size());
        for (const Hit& h : hits)
        {
            ids.push_back(h.Id);
        }
        totalHit += OverlapCount(ids, groundTruth[i]);
    }
    const double queryMs = NowMs() - tQuery;
    const double recall = static_cast<double>(totalHit) / (p.QueryCount * p.K);
    return BenchResult{ buildMs, queryMs, recall };
}

BenchResult RunHnswlib(const Dataset& ds, const std::vector<std::vector<float>>& normQueries,
                       const BenchParams& p, const std::vector<std::vector<std::size_t>>& groundTruth)
{
    hnswlib::InnerProductSpace space(p.Dim);
    hnswlib::HierarchicalNSW<float> index(&space, p.Count, p.MaxNeighbors, p.EfConstruction, 42);

    const double tBuild = NowMs();
    for (std::size_t id = 0; id < p.Count; ++id)
    {
        index.addPoint(ds.Normalized.data() + id * p.Dim, id);
    }
    const double buildMs = NowMs() - tBuild;

    index.setEf(p.Ef);
    std::size_t totalHit = 0;
    const double tQuery = NowMs();
    for (std::size_t i = 0; i < p.QueryCount; ++i)
    {
        auto result = index.searchKnn(normQueries[i].data(), p.K);
        std::vector<std::size_t> ids;
        ids.reserve(result.size());
        while (!result.empty())
        {
            ids.push_back(result.top().second);
            result.pop();
        }
        totalHit += OverlapCount(ids, groundTruth[i]);
    }
    const double queryMs = NowMs() - tQuery;
    const double recall = static_cast<double>(totalHit) / (p.QueryCount * p.K);
    return BenchResult{ buildMs, queryMs, recall };
}

std::string Fmt1(double value)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << value;
    return out.str();
}

std::string Fmt3(double value)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

void RunCase(const BenchParams& p)
{
    std::cout << "\n==== 用例：N=" << p.Count << " D=" << p.Dim << "  M=" << p.MaxNeighbors
              << " efC=" << p.EfConstruction << " ef=" << p.Ef << " k=" << p.K
              << " Q=" << p.QueryCount << " ====\n";

    const Dataset ds = GenerateDataset(p.Count, p.Dim, 7);
    const std::vector<std::vector<float>> rawQueries = GenerateQueries(p.QueryCount, p.Dim, 99);
    std::vector<std::vector<float>> normQueries = rawQueries;
    for (std::vector<float>& q : normQueries)
    {
        Beacon::L2Normalize(q.data(), q.size());
    }

    std::vector<std::vector<std::size_t>> groundTruth;
    groundTruth.reserve(p.QueryCount);
    for (std::size_t i = 0; i < p.QueryCount; ++i)
    {
        groundTruth.push_back(BruteForceTopK(normQueries[i], ds, p.K));
    }

    const BenchResult mine = RunBeacon(ds, rawQueries, p, groundTruth);
    const BenchResult theirs = RunHnswlib(ds, normQueries, p, groundTruth);

    const std::string recallCol = "recall@" + std::to_string(p.K);
    std::cout << "  " << std::left << std::setw(20) << "路径" << std::right
              << std::setw(12) << "build_ms" << std::setw(12) << recallCol
              << std::setw(12) << "ms/query" << "\n";
    std::cout << "  " << std::left << std::setw(20) << "Beacon" << std::right
              << std::setw(12) << Fmt1(mine.BuildMs)
              << std::setw(12) << Fmt3(mine.Recall)
              << std::setw(12) << Fmt1(mine.QueryMsTotal / p.QueryCount) << "\n";
    std::cout << "  " << std::left << std::setw(20) << "hnswlib" << std::right
              << std::setw(12) << Fmt1(theirs.BuildMs)
              << std::setw(12) << Fmt3(theirs.Recall)
              << std::setw(12) << Fmt1(theirs.QueryMsTotal / p.QueryCount) << "\n";
}

}  // namespace

int main(int argc, char** argv)
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

    if (argc >= 3)
    {
        const std::size_t count = static_cast<std::size_t>(std::stoull(argv[1]));
        const std::size_t dim = static_cast<std::size_t>(std::stoull(argv[2]));
        RunCase(BenchParams{ count, dim, 16, 200, 100, 10, 100 });
        return 0;
    }

    RunCase(BenchParams{ 30000, 64, 16, 200, 100, 10, 100 });
    RunCase(BenchParams{ 100000, 128, 16, 200, 100, 10, 100 });
    return 0;
}
