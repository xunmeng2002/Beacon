#include "mdbvec/metrics.h"

#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mdbvec {

#if defined(__AVX2__)

namespace {
// 8 个 float 横向累加
float hsum_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
}  // namespace

float dot_product(const float* a, const float* b, std::size_t n) {
  __m256 sum = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
  }
  float result = hsum_ps(sum);
  for (; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

#else  // 标量回退

float dot_product(const float* a, const float* b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

#endif

float l2_norm(const float* vec, std::size_t n) {
  return std::sqrt(dot_product(vec, vec, n));
}

void l2_normalize(float* vec, std::size_t n) {
  const float norm = l2_norm(vec, n);
  if (norm > 0.0f && std::isfinite(norm)) {
    const float inv = 1.0f / norm;
    for (std::size_t i = 0; i < n; ++i) {
      vec[i] *= inv;
    }
  }
}

}  // namespace mdbvec
