#include "echo/distance.h"

#include <cassert>
#include <cmath>

namespace echo {

float l2_squared(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float dot(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void normalize(std::span<float> v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    if (norm_sq == 0.0f) return;
    const float inv = 1.0f / std::sqrt(norm_sq);
    for (float& x : v) x *= inv;
}

float distance(Metric metric, std::span<const float> a, std::span<const float> b) {
    switch (metric) {
        case Metric::L2:
            return l2_squared(a, b);
        case Metric::Cosine:
            return 1.0f - dot(a, b);
    }
    return 0.0f;  // unreachable; keeps compilers happy
}

}  // namespace echo
