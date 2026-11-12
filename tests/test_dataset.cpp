#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "echo/dataset.h"

namespace {
const std::string kData = ECHO_TEST_DATA_DIR;
}

TEST(Dataset, RejectsZeroDimension) {
    EXPECT_THROW(echo::Dataset(0), std::invalid_argument);
}

TEST(Dataset, AddStoresRowsContiguouslyAndReturnsIndices) {
    echo::Dataset ds(3);
    std::vector<float> a{1, 2, 3}, b{4, 5, 6};
    EXPECT_EQ(ds.add(a), 0u);
    EXPECT_EQ(ds.add(b), 1u);
    ASSERT_EQ(ds.size(), 2u);
    // Row 1 starts right after row 0 in memory: that's the point of the layout.
    EXPECT_EQ(ds.row(1).data(), ds.row(0).data() + 3);
    EXPECT_FLOAT_EQ(ds.row(1)[2], 6.0f);
}

TEST(Dataset, AddRejectsWrongLength) {
    echo::Dataset ds(3);
    std::vector<float> short_vec{1, 2};
    EXPECT_THROW(ds.add(short_vec), std::invalid_argument);
}

TEST(Dataset, LoadsTsvWithMetadataAndFeatures) {
    const auto ds = echo::Dataset::load_tsv(kData + "/tiny.tsv", 2);
    EXPECT_EQ(ds.size(), 4u);
    EXPECT_EQ(ds.dim(), 3u);
    EXPECT_EQ(ds.meta_columns(), (std::vector<std::string>{"id", "name"}));
    EXPECT_EQ(ds.feature_columns(), (std::vector<std::string>{"x", "y", "z"}));
    EXPECT_EQ(ds.meta(2)[1], "Gamma Song");
    EXPECT_FLOAT_EQ(ds.row(2)[1], -1.5f);
}

TEST(Dataset, FindIsCaseInsensitiveSubstring) {
    const auto ds = echo::Dataset::load_tsv(kData + "/tiny.tsv", 2);
    EXPECT_EQ(ds.find(1, "gamma"), (std::vector<std::uint32_t>{2}));
    EXPECT_EQ(ds.find(1, "SONG").size(), 4u);
    EXPECT_TRUE(ds.find(1, "nonexistent").empty());
}

TEST(Dataset, LoadRejectsNonNumericFeature) {
    EXPECT_THROW(echo::Dataset::load_tsv(kData + "/bad_number.tsv", 2), std::runtime_error);
}

TEST(Dataset, LoadRejectsRaggedRows) {
    EXPECT_THROW(echo::Dataset::load_tsv(kData + "/ragged.tsv", 2), std::runtime_error);
}

TEST(Dataset, LoadRejectsMissingFile) {
    EXPECT_THROW(echo::Dataset::load_tsv(kData + "/does_not_exist.tsv", 2), std::runtime_error);
}
