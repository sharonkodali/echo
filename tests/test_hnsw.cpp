#include <gtest/gtest.h>

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
