// Approximate nearest-neighbor search with a navigable graph (HNSW).
//
// Brute force checks every vector. HNSW instead links each vector to a few
// of its nearest neighbors, forming a graph, and answers a query by walking
// that graph toward the target: from wherever you are, step to whichever
// neighbor is closer, until nothing is. That visits a tiny fraction of the
// vectors and still finds almost all of the true nearest neighbors.
//
// Built in steps:
//   1. One layer (this file, for now): a "navigable small world" graph.
//   2. Several layers, like a skip list, so the walk starts with long jumps.
//   3. Recall measured against BruteForceIndex.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "echo/brute_force.h"
#include "echo/dataset.h"
#include "echo/distance.h"

namespace echo {

struct HnswParams {
    // Links each vector keeps to its neighbors. More links = higher recall
    // and more memory. Typical values: 8-48.
    std::size_t M = 16;

    // How many candidates to track while finding a new vector's neighbors.
    // Bigger = better graph, slower build.
    std::size_t ef_construction = 200;

    // How many candidates to track while answering a query. This is the main
    // speed/accuracy dial at query time; it is raised to k if smaller.
    std::size_t ef_search = 50;
};

class HnswIndex {
public:
    // Copies the vectors and builds the graph by inserting them one by one.
    // For Metric::Cosine the copies are normalized first, as in brute force.
    HnswIndex(const Dataset& data, Metric metric, HnswParams params = {});

    std::size_t size() const { return data_.size(); }
    std::size_t dim() const { return data_.dim(); }
    Metric metric() const { return metric_; }
    const HnswParams& params() const { return params_; }

    // The vectors that `node` links to. Exposed for tests and debugging.
    const std::vector<std::uint32_t>& neighbors(std::uint32_t node) const {
        return links_.at(node);
    }

private:
    // Inserts row `node` into the graph: finds its nearest already-inserted
    // vectors with a beam search and links to them in both directions.
    void insert(std::uint32_t node);

    // Beam search: starting at `entry`, repeatedly expand the closest
    // unexplored candidate, keeping the best `ef` vectors seen so far. Stops
    // when the closest remaining candidate is farther than the worst of
    // those `ef`, because nothing reachable from it is likely to help.
    // Returns the best `ef` found, closest first.
    std::vector<Neighbor> search_layer(std::span<const float> q, std::uint32_t entry,
                                       std::size_t ef) const;

    // Keeps only the `max_links` closest neighbors of `node`.
    void shrink_links(std::uint32_t node, std::size_t max_links);

    float dist(std::span<const float> q, std::uint32_t node) const {
        return distance(metric_, q, data_.row(node));
    }

    Dataset data_;
    Metric metric_;
    HnswParams params_;
    std::vector<std::vector<std::uint32_t>> links_;  // links_[i] = neighbors of row i
    std::uint32_t entry_point_ = 0;                  // where every search starts
};

}  // namespace echo
