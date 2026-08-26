// MdbVector（自研 HNSW）与 hnswlib 基准对比：同一数据/查询下比 build_ms / recall@k / ms-query
// 构建需 vcpkg toolchain（MDBVEC_ENABLE_BENCH=ON），见 CMakeLists 与 README。
#include "mdbvec/Metrics.h"
#include "mdbvec/VectorDb.h"

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
    std::size_t count;
    std::size_t dim;
    std::size_t m;
    std::size_t ef_construction;
    std::size_t ef;
    std::size_t k;
    std::size_t query_count;
};

// 数据 = 原样（喂 MdbVector，内部余弦归一化）+ L2 归一化副本（喂 hnswlib 与暴力基准）
struct Dataset
{
    std::size_t count;
    std::size_t dim;
    std::vector<float> raw;
    std::vector<float> normalized;
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
        mdbvec::L2Normalize(row.data(), dim);
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
std::vector<std::size_t> BruteForceTopK(const std::vector<float>& query_norm, const Dataset& ds,
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
    for (std::size_t id = 0; id < ds.count; ++id)
    {
        const float score = mdbvec::DotProduct(query_norm.data(),
                                               ds.normalized.data() + id * ds.dim, ds.dim);
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
    double build_ms;
    double query_ms_total;
    double recall;
};

BenchResult RunMdbVector(const Dataset& ds, const std::vector<std::vector<float>>& raw_queries,
                         const BenchParams& p, const std::vector<std::vector<std::size_t>>& ground_truth)
{
    using namespace mdbvec;
    VectorDb db(p.dim, Metric::kCosine);
    db.Reserve(p.count);
    std::vector<float> row(p.dim);
    for (std::size_t id = 0; id < p.count; ++id)
    {
        std::copy(ds.raw.data() + id * p.dim, ds.raw.data() + (id + 1) * p.dim, row.begin());
        db.Add(row, "");
    }

    const double t_build = NowMs();
    db.EnableIndex(p.m, p.ef_construction);
    const double build_ms = NowMs() - t_build;

    std::size_t total_hit = 0;
    const double t_query = NowMs();
    for (std::size_t i = 0; i < p.query_count; ++i)
    {
        const std::vector<Hit> hits = db.SearchIndexed(raw_queries[i], p.k, p.ef);
        std::vector<std::size_t> ids;
        ids.reserve(hits.size());
        for (const Hit& h : hits)
        {
            ids.push_back(h.id);
        }
        total_hit += OverlapCount(ids, ground_truth[i]);
    }
    const double query_ms = NowMs() - t_query;
    const double recall = static_cast<double>(total_hit) / (p.query_count * p.k);
    return BenchResult{ build_ms, query_ms, recall };
}

BenchResult RunHnswlib(const Dataset& ds, const std::vector<std::vector<float>>& norm_queries,
                       const BenchParams& p, const std::vector<std::vector<std::size_t>>& ground_truth)
{
    hnswlib::InnerProductSpace space(p.dim);
    hnswlib::HierarchicalNSW<float> index(&space, p.count, p.m, p.ef_construction, 42);

    const double t_build = NowMs();
    for (std::size_t id = 0; id < p.count; ++id)
    {
        index.addPoint(ds.normalized.data() + id * p.dim, id);
    }
    const double build_ms = NowMs() - t_build;

    index.setEf(p.ef);
    std::size_t total_hit = 0;
    const double t_query = NowMs();
    for (std::size_t i = 0; i < p.query_count; ++i)
    {
        auto result = index.searchKnn(norm_queries[i].data(), p.k);
        std::vector<std::size_t> ids;
        ids.reserve(result.size());
        while (!result.empty())
        {
            ids.push_back(result.top().second);
            result.pop();
        }
        total_hit += OverlapCount(ids, ground_truth[i]);
    }
    const double query_ms = NowMs() - t_query;
    const double recall = static_cast<double>(total_hit) / (p.query_count * p.k);
    return BenchResult{ build_ms, query_ms, recall };
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
    std::cout << "\n==== 用例：N=" << p.count << " D=" << p.dim << "  M=" << p.m
              << " efC=" << p.ef_construction << " ef=" << p.ef << " k=" << p.k
              << " Q=" << p.query_count << " ====\n";

    const Dataset ds = GenerateDataset(p.count, p.dim, 7);
    const std::vector<std::vector<float>> raw_queries = GenerateQueries(p.query_count, p.dim, 99);
    std::vector<std::vector<float>> norm_queries = raw_queries;
    for (std::vector<float>& q : norm_queries)
    {
        mdbvec::L2Normalize(q.data(), q.size());
    }

    std::vector<std::vector<std::size_t>> ground_truth;
    ground_truth.reserve(p.query_count);
    for (std::size_t i = 0; i < p.query_count; ++i)
    {
        ground_truth.push_back(BruteForceTopK(norm_queries[i], ds, p.k));
    }

    const BenchResult mine = RunMdbVector(ds, raw_queries, p, ground_truth);
    const BenchResult theirs = RunHnswlib(ds, norm_queries, p, ground_truth);

    const std::string recall_col = "recall@" + std::to_string(p.k);
    std::cout << "  " << std::left << std::setw(20) << "路径" << std::right
              << std::setw(12) << "build_ms" << std::setw(12) << recall_col
              << std::setw(12) << "ms/query" << "\n";
    std::cout << "  " << std::left << std::setw(20) << "mdbvec(自研HNSW)" << std::right
              << std::setw(12) << Fmt1(mine.build_ms)
              << std::setw(12) << Fmt3(mine.recall)
              << std::setw(12) << Fmt1(mine.query_ms_total / p.query_count) << "\n";
    std::cout << "  " << std::left << std::setw(20) << "hnswlib" << std::right
              << std::setw(12) << Fmt1(theirs.build_ms)
              << std::setw(12) << Fmt3(theirs.recall)
              << std::setw(12) << Fmt1(theirs.query_ms_total / p.query_count) << "\n";
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
