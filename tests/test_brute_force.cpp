#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "echo/brute_force.h"

using echo::BruteForceIndex;
using echo::Dataset;
using echo::Metric;
using echo::Neighbor;

namespace {

// Points on a number line at 0, 1, 2, ..., n-1 (2-D, second coordinate 0),
// so the right answer for any query is obvious by hand.
Dataset number_line(std::size_t n) {
    Dataset ds(2);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> v{static_cast<float>(i), 0.0f};
        ds.add(v);
    }
    return ds;
}

std::vector<std::uint32_t> indices(const std::vector<Neighbor>& ns) {
    std::vector<std::uint32_t> out;
    for (const auto& n : ns) out.push_back(n.index);
    return out;
}

}  // namespace

TEST(BruteForce, FindsKNearestInOrder) {
    const BruteForceIndex index(number_line(10), Metric::L2);
    std::vector<float> q{4.2f, 0.0f};
    // Closest to 4.2: 4 (0.2 away), 5 (0.8), 3 (1.2)
    EXPECT_EQ(indices(index.search(q, 3)), (std::vector<std::uint32_t>{4, 5, 3}));
}

TEST(BruteForce, ResultsAreSortedClosestFirst) {
    const BruteForceIndex index(number_line(50), Metric::L2);
    std::vector<float> q{17.3f, 0.0f};
    const auto r = index.search(q, 10);
    EXPECT_TRUE(std::is_sorted(r.begin(), r.end()));
}

TEST(BruteForce, KLargerThanDatasetReturnsEverything) {
    const BruteForceIndex index(number_line(3), Metric::L2);
    std::vector<float> q{0.0f, 0.0f};
    EXPECT_EQ(index.search(q, 100).size(), 3u);
}

TEST(BruteForce, KZeroReturnsNothing) {
    const BruteForceIndex index(number_line(3), Metric::L2);
    std::vector<float> q{0.0f, 0.0f};
    EXPECT_TRUE(index.search(q, 0).empty());
}

TEST(BruteForce, ExcludeSkipsTheQueryItself) {
    const BruteForceIndex index(number_line(10), Metric::L2);
    std::vector<float> q{5.0f, 0.0f};
    const auto r = index.search(q, 2, /*exclude=*/5);
    EXPECT_EQ(indices(r), (std::vector<std::uint32_t>{4, 6}));
}

TEST(BruteForce, TiesBrokenByIndexDeterministically) {
    const BruteForceIndex index(number_line(10), Metric::L2);
    std::vector<float> q{5.5f, 0.0f};  // 5 and 6 are exactly equally close
    EXPECT_EQ(indices(index.search(q, 2)), (std::vector<std::uint32_t>{5, 6}));
}

TEST(BruteForce, RejectsWrongQueryLength) {
    const BruteForceIndex index(number_line(3), Metric::L2);
    std::vector<float> q{1.0f};
    EXPECT_THROW(index.search(q, 1), std::invalid_argument);
}

TEST(BruteForce, CosineRanksByDirectionNotLength) {
    Dataset ds(2);
    std::vector<float> far_same_direction{100, 1}, near_other_direction{0, 2};
    ds.add(far_same_direction);
    ds.add(near_other_direction);
    std::vector<float> q{1, 0};
    // L2 says the short vector is closer; cosine says the same-direction one is.
    EXPECT_EQ(BruteForceIndex(ds, Metric::L2).search(q, 1)[0].index, 1u);
    EXPECT_EQ(BruteForceIndex(ds, Metric::Cosine).search(q, 1)[0].index, 0u);
}

TEST(BruteForce, CosineDoesNotModifyCallersDataOrQuery) {
    Dataset ds(2);
    std::vector<float> v{3, 4};
    ds.add(v);
    std::vector<float> q{6, 8};
    const BruteForceIndex index(ds, Metric::Cosine);
    index.search(q, 1);
    EXPECT_FLOAT_EQ(ds.row(0)[0], 3.0f);  // index normalized its own copy
    EXPECT_FLOAT_EQ(q[0], 6.0f);
}

// The heap-based top-k must agree with the simplest possible approach:
// compute every distance and sort. This is how HNSW will be checked too.
TEST(BruteForce, MatchesFullSortOnRandomData) {
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss;
    Dataset ds(8);
    std::vector<float> v(8);
    for (int i = 0; i < 500; ++i) {
        for (auto& x : v) x = gauss(rng);
        ds.add(v);
    }
    const BruteForceIndex index(ds, Metric::L2);
    for (int trial = 0; trial < 20; ++trial) {
        for (auto& x : v) x = gauss(rng);
        std::vector<Neighbor> all;
        for (std::uint32_t i = 0; i < ds.size(); ++i) all.push_back({i, echo::l2_squared(v, ds.row(i))});
        std::sort(all.begin(), all.end());
        all.resize(10);
        EXPECT_EQ(indices(index.search(v, 10)), indices(all));
    }
}

TEST(Recall, PerfectPartialAndEmpty) {
    std::vector<Neighbor> truth{{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    EXPECT_DOUBLE_EQ(echo::recall(truth, truth), 1.0);
    std::vector<Neighbor> half{{1, 0}, {9, 0}, {3, 0}, {8, 0}};
    EXPECT_DOUBLE_EQ(echo::recall(half, truth), 0.5);
    EXPECT_DOUBLE_EQ(echo::recall({}, truth), 0.0);
    EXPECT_DOUBLE_EQ(echo::recall(half, {}), 1.0);
}

TEST(Recall, IgnoresOrder) {
    std::vector<Neighbor> truth{{1, 0}, {2, 0}}, reversed{{2, 0}, {1, 0}};
    EXPECT_DOUBLE_EQ(echo::recall(reversed, truth), 1.0);
}
