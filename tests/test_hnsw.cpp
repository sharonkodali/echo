#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
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
