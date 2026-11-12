#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "echo/distance.h"

using echo::Metric;

TEST(L2Squared, ZeroForIdenticalVectors) {
    std::vector<float> a{1.5f, -2.0f, 3.25f};
    EXPECT_FLOAT_EQ(echo::l2_squared(a, a), 0.0f);
}

TEST(L2Squared, MatchesHandComputedValue) {
    // (1-4)^2 + (2-6)^2 = 9 + 16 = 25  (the 3-4-5 triangle, squared)
    std::vector<float> a{1, 2}, b{4, 6};
    EXPECT_FLOAT_EQ(echo::l2_squared(a, b), 25.0f);
}

TEST(L2Squared, IsSymmetric) {
    std::vector<float> a{0.1f, 0.7f, -3.0f}, b{2.0f, -1.0f, 0.5f};
    EXPECT_FLOAT_EQ(echo::l2_squared(a, b), echo::l2_squared(b, a));
}

TEST(Dot, MatchesHandComputedValue) {
    std::vector<float> a{1, 2, 3}, b{4, -5, 6};
    EXPECT_FLOAT_EQ(echo::dot(a, b), 4.0f - 10.0f + 18.0f);
}

TEST(Normalize, ProducesUnitLength) {
    std::vector<float> v{3, 4};
    echo::normalize(v);
    EXPECT_NEAR(v[0], 0.6f, 1e-6f);
    EXPECT_NEAR(v[1], 0.8f, 1e-6f);
    EXPECT_NEAR(echo::dot(v, v), 1.0f, 1e-6f);
}

TEST(Normalize, LeavesZeroVectorUnchanged) {
    std::vector<float> v{0, 0, 0};
    echo::normalize(v);  // must not divide by zero and produce NaN
    for (float x : v) EXPECT_EQ(x, 0.0f);
}

TEST(CosineDistance, IgnoresVectorLength) {
    // Same direction, different lengths: cosine says identical, L2 doesn't.
    std::vector<float> a{1, 2, 3}, b{10, 20, 30};
    echo::normalize(a);
    echo::normalize(b);
    EXPECT_NEAR(echo::distance(Metric::Cosine, a, b), 0.0f, 1e-6f);
}

TEST(CosineDistance, OrthogonalIsOneOppositeIsTwo) {
    std::vector<float> x{1, 0}, y{0, 1}, minus_x{-1, 0};
    EXPECT_NEAR(echo::distance(Metric::Cosine, x, y), 1.0f, 1e-6f);
    EXPECT_NEAR(echo::distance(Metric::Cosine, x, minus_x), 2.0f, 1e-6f);
}

TEST(Distance, L2MetricUsesSquaredEuclidean) {
    std::vector<float> a{1, 2}, b{4, 6};
    EXPECT_FLOAT_EQ(echo::distance(Metric::L2, a, b), 25.0f);
}
