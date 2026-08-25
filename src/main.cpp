#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mdbvec/vector_db.h"

class Stopwatch {
public:
  void start() { begin_ = std::chrono::steady_clock::now(); }
  double elapsed_ms() const {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - begin_).count();
  }

private:
  std::chrono::steady_clock::time_point begin_;
};

static void demo_small();
static void demo_large();

int main() {
  demo_small();
  demo_large();
  return 0;
}

// 3 维余弦检索 + 持久化往返
static void demo_small() {
  using namespace mdbvec;
  std::cout << "==== 演示 1：3 维余弦检索 ====\n";

  VectorDb db(3, Metric::Cosine);
  db.add({1, 0, 0}, "x 轴");
  db.add({0, 1, 0}, "y 轴");
  db.add({0, 0, 1}, "z 轴");
  db.add({0.9f, 0.1f, 0}, "接近 x 轴");

  const auto hits = db.search({0.95f, 0.05f, 0}, 2);
  for (const auto& h : hits) {
    std::cout << "  top id=" << h.id << " score=" << h.score
              << " meta=" << db.metadata(h.id) << "\n";
  }

  const std::string path = "demo_small.mdbv";
  std::cout << "  持久化到 " << path << " ...\n";
  if (!db.save(path)) {
    std::cerr << "  保存失败\n";
    return;
  }
  VectorDb loaded(3, Metric::Cosine);
  if (!loaded.load(path)) {
    std::cerr << "  加载失败\n";
    return;
  }
  const auto hits2 = loaded.search({0.95f, 0.05f, 0}, 1);
  std::cout << "  重载后 top-1 id=" << hits2[0].id
            << " meta=" << loaded.metadata(hits2[0].id) << "\n";
}

// 10k 条 64 维暴力检索，量测延迟
static void demo_large() {
  using namespace mdbvec;
  std::cout << "==== 演示 2：10k x 64 维暴力检索延迟 ====\n";

  const std::size_t count = 10000;
  const std::size_t dim = 64;
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  VectorDb db(dim, Metric::Cosine);
  std::vector<float> vec(dim);
  for (std::size_t i = 0; i < count; ++i) {
    for (auto& x : vec) x = dist(rng);
    db.add(vec, "vec-" + std::to_string(i));
  }

  for (auto& x : vec) x = dist(rng);
  Stopwatch sw;
  sw.start();
  const auto hits = db.search(vec, 5);
  const double ms = sw.elapsed_ms();

  std::cout << "  数据量=" << db.count() << " 维度=" << dim
            << " 检索耗时=" << ms << " ms\n";
  std::cout << "  top-1 id=" << hits[0].id << " score=" << hits[0].score << "\n";
}
