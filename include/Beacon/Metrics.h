// 距离度量：点积与 L2 归一化
#pragma once
#include <cstddef>

namespace beacon {

// a 与 b 的点积（a·b），x86_64 上走 AVX2，其余平台标量回退
float DotProduct(const float* a, const float* b, std::size_t n);

// 向量的 L2 范数
float L2Norm(const float* vec, std::size_t n);

// 原地 L2 归一化（除以模长）；零向量保持不变
void L2Normalize(float* vec, std::size_t n);

}  // namespace beacon
