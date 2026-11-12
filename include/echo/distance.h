// Distance functions between vectors.
//
// Every search in Echo boils down to "how far apart are these two points?",
// called millions of times. These functions are deliberately simple loops
// over contiguous floats: the compiler can auto-vectorize them (process
// several floats per instruction), and later weeks can hand-tune them with
// SIMD and measure the difference.
#pragma once

#include <cstddef>
#include <span>

namespace echo {

// How "closeness" is measured.
//   L2:     straight-line (Euclidean) distance. Sensitive to vector length.
//   Cosine: angle between vectors, ignoring length. Echo implements it as
//           1 - dot(a, b) on vectors normalized to length 1, so a smaller
//           value always means "more similar", same as L2.
enum class Metric { L2, Cosine };

// Squared Euclidean distance. Skipping the square root is safe for ranking:
// if a is closer than b, a^2 is smaller than b^2, so the order is unchanged.
float l2_squared(std::span<const float> a, std::span<const float> b);

// Sum of element-wise products.
float dot(std::span<const float> a, std::span<const float> b);

// Scales v in place to length 1. A zero vector is left unchanged (it has no
// direction, so there is nothing sensible to scale it to).
void normalize(std::span<float> v);

// Distance under the given metric. For Metric::Cosine, both inputs must
// already be normalized.
float distance(Metric metric, std::span<const float> a, std::span<const float> b);

}  // namespace echo
