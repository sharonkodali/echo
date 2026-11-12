// A set of fixed-length vectors stored in one contiguous block of memory.
//
// Why one flat array instead of std::vector<std::vector<float>>? Each inner
// vector would be a separate heap allocation scattered around memory, so
// scanning them would keep missing the CPU cache. A flat array lets the CPU
// stream through vectors back to back, which is a big part of why C++ search
// libraries are fast.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace echo {

class Dataset {
public:
    explicit Dataset(std::size_t dim) : dim_(dim) {
        if (dim == 0) throw std::invalid_argument("Dataset dimension must be > 0");
    }

    std::size_t dim() const { return dim_; }
    std::size_t size() const { return values_.size() / dim_; }
    bool empty() const { return values_.empty(); }

    // Appends one vector (must have exactly dim() values) and returns its index.
    std::uint32_t add(std::span<const float> v, std::vector<std::string> meta = {});

    std::span<const float> row(std::size_t i) const {
        return {values_.data() + i * dim_, dim_};
    }
    std::span<float> row(std::size_t i) { return {values_.data() + i * dim_, dim_}; }

    // Optional text attached to each vector (for songs: id, name, artist, genre).
    const std::vector<std::string>& meta(std::size_t i) const { return meta_.at(i); }
    const std::vector<std::string>& meta_columns() const { return meta_columns_; }
    const std::vector<std::string>& feature_columns() const { return feature_columns_; }

    // Returns the indices of rows whose given metadata column contains
    // `needle`, ignoring case. Used to look songs up by name.
    std::vector<std::uint32_t> find(std::size_t meta_col, const std::string& needle) const;

    // Loads a tab-separated file: a header row, then one row per vector. The
    // first `num_meta_cols` columns are text; every remaining column is a float.
    static Dataset load_tsv(const std::string& path, std::size_t num_meta_cols);

private:
    std::size_t dim_;
    std::vector<float> values_;  // size() * dim_ floats, row after row
    std::vector<std::vector<std::string>> meta_;
    std::vector<std::string> meta_columns_;
    std::vector<std::string> feature_columns_;
};

}  // namespace echo
