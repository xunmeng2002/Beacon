#include "MdbVector/Metrics.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <vector>

using namespace mdbvec;

TEST(MetricsTest, DotProductMatchesNaiveReference)
{
    // 已知值：{1,2,3}·{4,5,6} = 4+10+18 = 32
    const std::vector<float> a = { 1, 2, 3 };
    const std::vector<float> b = { 4, 5, 6 };
    EXPECT_FLOAT_EQ(DotProduct(a.data(), b.data(), 3), 32.0f);

    // 随机向量 vs 朴素循环参考：校验 AVX2 与标量回退路径结果一致
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (const std::size_t n : { 1u, 2u, 3u, 16u, 17u, 64u, 65u })
    {
        std::vector<float> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = dist(rng);
            y[i] = dist(rng);
        }
        float reference = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            reference += x[i] * y[i];
        }
        EXPECT_NEAR(DotProduct(x.data(), y.data(), n), reference, 1e-4f);
    }
}

TEST(MetricsTest, L2NormComputesVectorLength)
{
    const std::vector<float> v = { 3, 4 };
    EXPECT_FLOAT_EQ(L2Norm(v.data(), 2), 5.0f);
}

TEST(MetricsTest, L2NormalizeScalesToUnitLengthAndKeepsZeroVector)
{
    std::vector<float> v = { 3, 4 };
    L2Normalize(v.data(), 2);
    EXPECT_NEAR(v[0], 0.6f, 1e-6f);
    EXPECT_NEAR(v[1], 0.8f, 1e-6f);
    EXPECT_NEAR(L2Norm(v.data(), 2), 1.0f, 1e-6f);

    // 零向量保持不变（防除零回归）
    std::vector<float> zero = { 0, 0, 0 };
    L2Normalize(zero.data(), 3);
    for (float x : zero)
    {
        EXPECT_FLOAT_EQ(x, 0.0f);
    }
}
