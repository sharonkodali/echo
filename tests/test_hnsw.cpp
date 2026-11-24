#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

#include "echo/hnsw.h"

using echo::Dataset;
using echo::HnswIndex;
using echo::HnswParams;
using echo::Metric;

namespace {

Dataset number_line(std::size_t n) {
    Dataset ds(2);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> v{static_cast<float>(i), 0.0f};
        ds.add(v);
    }
    return ds;
}


Dataset random_data(std::size_t n, std::size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss;
    Dataset ds(dim);
    std::vector<float> v(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& x : v) x = gauss(rng);
        ds.add(v);
    }
    return ds;
}

}  // namespace

TEST(Hnsw, RejectsBadParams) {
    EXPECT_THROW(HnswIndex(number_line(5), Metric::L2, HnswParams{.M = 1}), std::invalid_argument);
    EXPECT_THROW(HnswIndex(number_line(5), Metric::L2, HnswParams{.M = 16, .ef_construction = 8}),
                 std::invalid_argument);
}

TEST(Hnsw, ReportsSizeAndDim) {
    const HnswIndex index(number_line(7), Metric::L2);
    EXPECT_EQ(index.size(), 7u);
    EXPECT_EQ(index.dim(), 2u);
}

TEST(Hnsw, EveryVectorGetsLinks) {
    const HnswIndex index(number_line(100), Metric::L2, HnswParams{.M = 4, .ef_construction = 16});
    for (std::uint32_t i = 0; i < index.size(); ++i) {
        EXPECT_FALSE(index.neighbors(i).empty()) << "row " << i << " is unreachable";
    }
}

TEST(Hnsw, LinksStayWithinLimitAndNeverPointToSelf) {
    const HnswParams p{.M = 4, .ef_construction = 16};
    const HnswIndex index(number_line(200), Metric::L2, p);
    for (std::uint32_t i = 0; i < index.size(); ++i) {
        const auto& ls = index.neighbors(i);
        EXPECT_LE(ls.size(), 2 * p.M);
        EXPECT_EQ(std::count(ls.begin(), ls.end(), i), 0);
    }
}

TEST(Hnsw, LinksPointToNearbyVectors) {
    // On a number line, each point's links should be its close neighbors.
    const HnswIndex index(number_line(50), Metric::L2, HnswParams{.M = 2, .ef_construction = 50});
    for (const std::uint32_t nb : index.neighbors(25)) {
        EXPECT_LE(std::abs(static_cast<int>(nb) - 25), 2);
    }
}

TEST(Hnsw, FindsExactAnswerOnSmallData) {
    const HnswIndex index(number_line(10), Metric::L2);
    std::vector<float> q{4.2f, 0.0f};
    const auto r = index.search(q, 3);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0].index, 4u);
    EXPECT_EQ(r[1].index, 5u);
    EXPECT_EQ(r[2].index, 3u);
}

TEST(Hnsw, ExcludeSkipsTheQueryItselfAndStillReturnsK) {
    const HnswIndex index(number_line(20), Metric::L2);
    std::vector<float> q{5.0f, 0.0f};
    const auto r = index.search(q, 2, /*exclude=*/5);
    ASSERT_EQ(r.size(), 2u);
    EXPECT_NE(r[0].index, 5u);
    EXPECT_NE(r[1].index, 5u);
}

TEST(Hnsw, EdgeCases) {
    const HnswIndex one(number_line(1), Metric::L2);
    std::vector<float> q{3.0f, 0.0f};
    EXPECT_EQ(one.search(q, 5).size(), 1u);
    EXPECT_TRUE(one.search(q, 0).empty());
    std::vector<float> wrong{1.0f};
    EXPECT_THROW(one.search(wrong, 1), std::invalid_argument);
}

TEST(Hnsw, ResultsAreSortedClosestFirst) {
    const auto data = random_data(1000, 8, 3);
    const HnswIndex index(data, Metric::L2);
    const auto r = index.search(data.row(0), 10);
    EXPECT_TRUE(std::is_sorted(r.begin(), r.end()));
}

// The real test: on random data, HNSW must find almost the same neighbors
// as the exact brute-force answer key.
TEST(Hnsw, HighRecallAgainstBruteForce) {
    for (const Metric metric : {Metric::L2, Metric::Cosine}) {
        const auto data = random_data(2000, 16, 11);
        const auto queries = random_data(100, 16, 12);
        const echo::BruteForceIndex exact(data, metric);
        const HnswIndex approx(data, metric, HnswParams{.M = 16, .ef_construction = 100});

        double total = 0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            total += echo::recall(approx.search(queries.row(i), 10, -1, 64),
                                  exact.search(queries.row(i), 10));
        }
        EXPECT_GE(total / static_cast<double>(queries.size()), 0.95)
            << (metric == Metric::L2 ? "L2" : "cosine");
    }
}

TEST(Hnsw, LargerEfGivesAtLeastAsMuchRecall) {
    const auto data = random_data(2000, 16, 21);
    const auto queries = random_data(50, 16, 22);
    const echo::BruteForceIndex exact(data, Metric::L2);
    const HnswIndex approx(data, Metric::L2, HnswParams{.M = 8, .ef_construction = 64});
    auto avg_recall = [&](std::size_t ef) {
        double t = 0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            t += echo::recall(approx.search(queries.row(i), 10, -1, ef),
                              exact.search(queries.row(i), 10));
        }
        return t / static_cast<double>(queries.size());
    };
    EXPECT_GE(avg_recall(200), avg_recall(10));
}
