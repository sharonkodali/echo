#include "echo/brute_force.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace echo {

BruteForceIndex::BruteForceIndex(const Dataset& data, Metric metric)
    : data_(data), metric_(metric) {
    if (metric_ == Metric::Cosine) {
        for (std::size_t i = 0; i < data_.size(); ++i) normalize(data_.row(i));
    }
}

std::vector<Neighbor> BruteForceIndex::search(std::span<const float> query, std::size_t k,
                                              std::int64_t exclude) const {
    if (query.size() != data_.dim()) {
        throw std::invalid_argument("query has " + std::to_string(query.size()) +
                                    " values, index expects " + std::to_string(data_.dim()));
    }
    if (k == 0) return {};

    std::vector<float> q(query.begin(), query.end());
    if (metric_ == Metric::Cosine) normalize(q);

    // Keep the k best candidates in a max-heap: the top is the *worst* of the
    // current best k, so each new vector only has to beat that one. This is
    // O(n log k) instead of sorting all n distances, O(n log n).
    std::priority_queue<Neighbor> best;
    for (std::size_t i = 0; i < data_.size(); ++i) {
        if (static_cast<std::int64_t>(i) == exclude) continue;
        const Neighbor cand{static_cast<std::uint32_t>(i), distance(metric_, q, data_.row(i))};
        if (best.size() < k) {
            best.push(cand);
        } else if (cand < best.top()) {
            best.pop();
            best.push(cand);
        }
    }

    std::vector<Neighbor> out;
    out.reserve(best.size());
    while (!best.empty()) {
        out.push_back(best.top());
        best.pop();
    }
    std::reverse(out.begin(), out.end());  // heap pops worst first; flip to closest first
    return out;
}

double recall(const std::vector<Neighbor>& found, const std::vector<Neighbor>& truth) {
    if (truth.empty()) return 1.0;
    std::unordered_set<std::uint32_t> want;
    for (const auto& n : truth) want.insert(n.index);
    std::size_t hits = 0;
    for (const auto& n : found) hits += want.count(n.index);
    return static_cast<double>(hits) / static_cast<double>(truth.size());
}

}  // namespace echo
