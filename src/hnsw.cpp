#include "echo/hnsw.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace echo {

namespace {

// Links allowed per vector. HNSW lets the bottom layer keep 2*M: it holds
// every vector, so extra links there buy recall cheaply.
std::size_t max_links(const HnswParams& p) { return 2 * p.M; }

// Orders a priority_queue so the *closest* neighbor is on top.
struct Farther {
    bool operator()(const Neighbor& a, const Neighbor& b) const { return b < a; }
};

}  // namespace

HnswIndex::HnswIndex(const Dataset& data, Metric metric, HnswParams params)
    : data_(data), metric_(metric), params_(params) {
    if (params_.M < 2) throw std::invalid_argument("HNSW M must be at least 2");
    if (params_.ef_construction < params_.M) {
        throw std::invalid_argument("HNSW ef_construction must be at least M");
    }
    if (metric_ == Metric::Cosine) {
        for (std::size_t i = 0; i < data_.size(); ++i) normalize(data_.row(i));
    }

    links_.resize(data_.size());
    for (std::uint32_t i = 0; i < data_.size(); ++i) insert(i);
}

void HnswIndex::insert(std::uint32_t node) {
    if (node == 0) return;  // the first vector has nobody to link to yet

    const auto q = data_.row(node);
    auto found = search_layer(q, entry_point_, params_.ef_construction);
    if (found.size() > params_.M) found.resize(params_.M);

    for (const auto& n : found) {
        links_[node].push_back(n.index);
        // Links go both ways, otherwise later searches could reach the old
        // vector but never step from it to the new one.
        links_[n.index].push_back(node);
        if (links_[n.index].size() > max_links(params_)) shrink_links(n.index, max_links(params_));
    }
}

std::vector<Neighbor> HnswIndex::search_layer(std::span<const float> q, std::uint32_t entry,
                                              std::size_t ef) const {
    // visited[i] marks rows already measured, so no row is checked twice.
    std::vector<bool> visited(data_.size(), false);

    // candidates: min-heap, closest unexplored vector on top.
    // best:       max-heap of the best `ef` so far, worst on top.
    std::priority_queue<Neighbor, std::vector<Neighbor>, Farther> candidates;
    std::priority_queue<Neighbor> best;

    const Neighbor start{entry, dist(q, entry)};
    visited[entry] = true;
    candidates.push(start);
    best.push(start);

    while (!candidates.empty()) {
        const Neighbor cur = candidates.top();
        if (best.size() >= ef && best.top() < cur) break;  // can't improve anymore
        candidates.pop();

        for (const std::uint32_t nb : links_[cur.index]) {
            if (visited[nb]) continue;
            visited[nb] = true;
            const Neighbor cand{nb, dist(q, nb)};
            if (best.size() < ef || cand < best.top()) {
                candidates.push(cand);
                best.push(cand);
                if (best.size() > ef) best.pop();
            }
        }
    }

    std::vector<Neighbor> out;
    out.reserve(best.size());
    while (!best.empty()) {
        out.push_back(best.top());
        best.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
}

void HnswIndex::shrink_links(std::uint32_t node, std::size_t max) {
    auto& ls = links_[node];
    const auto v = data_.row(node);
    std::vector<Neighbor> ranked;
    ranked.reserve(ls.size());
    for (const std::uint32_t nb : ls) ranked.push_back({nb, dist(v, nb)});
    std::sort(ranked.begin(), ranked.end());
    ls.clear();
    for (std::size_t i = 0; i < max && i < ranked.size(); ++i) ls.push_back(ranked[i].index);
}

}  // namespace echo
