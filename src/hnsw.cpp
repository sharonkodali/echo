#include "echo/hnsw.h"

#include <stdexcept>

namespace echo {

HnswIndex::HnswIndex(const Dataset& data, Metric metric, HnswParams params)
    : data_(data), metric_(metric), params_(params) {
    if (params_.M < 2) throw std::invalid_argument("HNSW M must be at least 2");
    if (params_.ef_construction < params_.M) {
        throw std::invalid_argument("HNSW ef_construction must be at least M");
    }
    if (metric_ == Metric::Cosine) {
        for (std::size_t i = 0; i < data_.size(); ++i) normalize(data_.row(i));
    }
}

}  // namespace echo
