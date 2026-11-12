// Exact nearest-neighbor search by checking every vector.
//
// This is the "answer key" for the whole project. It is too slow to scale
// (the work grows with every vector added), but it is always exactly right.
// When HNSW arrives in week 2, its results are scored against this index:
// recall@k = what fraction of the true k nearest neighbors HNSW found.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "echo/dataset.h"
#include "echo/distance.h"

namespace echo {

struct Neighbor {
    std::uint32_t index;  // row in the dataset
    float distance;       // smaller = more similar

    bool operator<(const Neighbor& other) const {
        // Ties are broken by index so results are deterministic, which makes
        // them testable and comparable across runs.
        return distance < other.distance || (distance == other.distance && index < other.index);
    }
};

class BruteForceIndex {
public:
    // Copies the vectors. For Metric::Cosine the copies are normalized once
    // up front, so each query only needs a dot product.
    BruteForceIndex(const Dataset& data, Metric metric);

    // Returns up to k nearest neighbors of `query`, closest first. If
    // `exclude` is a valid row, that row is skipped (handy for "songs like
    // this one", where the song itself would otherwise be the top match).
    std::vector<Neighbor> search(std::span<const float> query, std::size_t k,
                                 std::int64_t exclude = -1) const;

    std::size_t size() const { return data_.size(); }
    std::size_t dim() const { return data_.dim(); }
    Metric metric() const { return metric_; }

private:
    Dataset data_;
    Metric metric_;
};

// Fraction of `truth` (the exact answer) that also appears in `found`.
// Order doesn't matter: recall only asks "did you find the right items?"
double recall(const std::vector<Neighbor>& found, const std::vector<Neighbor>& truth);

}  // namespace echo
