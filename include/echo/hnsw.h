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

private:
    Dataset data_;
    Metric metric_;
    HnswParams params_;
};

}  // namespace echo
