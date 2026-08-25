#include "mdbvec/vector_db.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <queue>

#include "mdbvec/metrics.h"

namespace mdbvec {

namespace {
constexpr char kMagic[4] = {'M', 'D', 'B', 'V'};
constexpr std::uint32_t kFormatVersion = 1;

// 优先队列按 score 最小堆组织（堆顶是最差候选）
struct MinScoreFirst {
  bool operator()(const Hit& a, const Hit& b) const { return a.score > b.score; }
};

std::uint8_t metric_to_u8(Metric m) { return m == Metric::Cosine ? 0u : 1u; }
Metric u8_to_metric(std::uint8_t v) { return v == 0 ? Metric::Cosine : Metric::InnerProduct; }

bool write_str(std::ofstream& out, const std::string& s) {
  const auto len = static_cast<std::uint64_t>(s.size());
  out.write(reinterpret_cast<const char*>(&len), sizeof(len));
  if (!s.empty()) {
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
  }
  return static_cast<bool>(out);
}

bool read_str(std::ifstream& in, std::string& s) {
  std::uint64_t len = 0;
  in.read(reinterpret_cast<char*>(&len), sizeof(len));
  if (!in) return false;
  s.resize(static_cast<std::size_t>(len));
  if (len > 0) {
    in.read(s.data(), static_cast<std::streamsize>(s.size()));
  }
  return static_cast<bool>(in);
}
}  // namespace

VectorDb::VectorDb(std::size_t dim, Metric metric) : table_(dim, metric) {}

std::size_t VectorDb::add(const std::vector<float>& vec, const std::string& meta) {
  return table_.add(vec, meta);
}

std::vector<Hit> VectorDb::search(const std::vector<float>& query, std::size_t k) const {
  std::vector<Hit> result;
  const std::size_t n = table_.count();
  if (n == 0 || k == 0 || query.size() != table_.dim()) {
    return result;
  }

  std::vector<float> q = query;
  if (table_.metric() == Metric::Cosine) {
    l2_normalize(q.data(), q.size());
  }

  const std::size_t cap = std::min(k, n);
  std::priority_queue<Hit, std::vector<Hit>, MinScoreFirst> heap;
  for (std::size_t id = 0; id < n; ++id) {
    const float score = dot_product(q.data(), table_.vector(id), table_.dim());
    if (heap.size() < cap) {
      heap.push(Hit{id, score});
    } else if (score > heap.top().score) {
      heap.pop();
      heap.push(Hit{id, score});
    }
  }

  result.reserve(heap.size());
  while (!heap.empty()) {
    result.push_back(heap.top());
    heap.pop();
  }
  std::reverse(result.begin(), result.end());  // 升序 -> 降序（最好的在前）
  return result;
}

std::size_t VectorDb::count() const { return table_.count(); }
std::size_t VectorDb::dim() const { return table_.dim(); }
const std::string& VectorDb::metadata(std::size_t id) const { return table_.metadata(id); }

void VectorDb::clear() { table_ = VectorTable(); }

bool VectorDb::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;

  const auto dim = static_cast<std::uint64_t>(table_.dim());
  const auto count = static_cast<std::uint64_t>(table_.count());
  const auto metric = metric_to_u8(table_.metric());
  const auto version = kFormatVersion;

  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  if (!out) return false;

  if (count > 0) {
    const std::uint64_t float_count = dim * count;
    out.write(reinterpret_cast<const char*>(table_.data()),
              static_cast<std::streamsize>(float_count * sizeof(float)));
    if (!out) return false;
    for (std::size_t id = 0; id < table_.count(); ++id) {
      if (!write_str(out, table_.metadata(id))) return false;
    }
  }
  return true;
}

bool VectorDb::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  char magic[4];
  in.read(magic, 4);
  if (!in || std::string(magic, 4) != std::string(kMagic, 4)) return false;

  std::uint32_t version = 0;
  std::uint64_t dim = 0, count = 0;
  std::uint8_t metric = 0;
  in.read(reinterpret_cast<char*>(&version), sizeof(version));
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  in.read(reinterpret_cast<char*>(&metric), sizeof(metric));
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!in || version != kFormatVersion) return false;

  std::vector<float> data;
  std::vector<std::string> metadata;
  if (count > 0) {
    const std::uint64_t float_count = dim * count;
    data.resize(static_cast<std::size_t>(float_count));
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(float_count * sizeof(float)));
    if (!in) return false;
    metadata.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
      std::string s;
      if (!read_str(in, s)) return false;
      metadata.push_back(std::move(s));
    }
  }

  table_.set_data(static_cast<std::size_t>(dim), u8_to_metric(metric),
                  std::move(data), std::move(metadata));
  return true;
}

}  // namespace mdbvec
