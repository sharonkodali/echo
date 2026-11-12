// bench_bruteforce: measures how fast exact search is, and how it scales.
//
//   ./build/bench_bruteforce data/songs.tsv
//
// Brute force does work proportional to (number of vectors x dimensions)
// per query. This benchmark runs it on the real song data, then on random
// data at growing sizes and dimensions, to show where it stops being fast.
// That crossover is the reason HNSW exists, and week 3 plots HNSW on the
// same axes.
#include <chrono>
#include <cstdio>
#include <random>
#include <string>

#include "echo/brute_force.h"
#include "echo/dataset.h"

namespace {

echo::Dataset random_dataset(std::size_t n, std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    echo::Dataset ds(dim);
    std::vector<float> v(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& x : v) x = gauss(rng);
        ds.add(v);
    }
    return ds;
}

// Average milliseconds per query over `queries` queries drawn from the data.
double ms_per_query(const echo::Dataset& data, std::size_t queries, std::size_t k) {
    const echo::BruteForceIndex index(data, echo::Metric::L2);
    std::size_t checksum = 0;  // stops the compiler optimizing the loop away
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t q = 0; q < queries; ++q) {
        const auto r = index.search(data.row((q * 7919) % data.size()), k);
        checksum += r.front().index;
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (checksum == static_cast<std::size_t>(-1)) std::puts("");
    return std::chrono::duration<double, std::milli>(elapsed).count() / static_cast<double>(queries);
}

}  // namespace

int main(int argc, char** argv) {
    constexpr std::size_t k = 10;
    std::printf("%-28s %10s %6s %12s %12s\n", "dataset", "vectors", "dim", "ms/query", "queries/s");

    if (argc > 1) {
        const auto songs = echo::Dataset::load_tsv(argv[1], 4);
        const double ms = ms_per_query(songs, 500, k);
        std::printf("%-28s %10zu %6zu %12.4f %12.0f\n", "songs (real)", songs.size(), songs.dim(), ms,
                    1000.0 / ms);
    }

    std::mt19937 rng(42);
    for (std::size_t dim : {16, 128}) {
        for (std::size_t n : {10'000, 100'000, 1'000'000}) {
            if (dim == 128 && n == 1'000'000) continue;  // ~512 MB; skip on laptops
            const auto data = random_dataset(n, dim, rng);
            const double ms = ms_per_query(data, n >= 1'000'000 ? 20 : 100, k);
            std::printf("%-28s %10zu %6zu %12.4f %12.0f\n", "random gaussian", n, dim, ms, 1000.0 / ms);
        }
    }
    std::puts("\nBrute force cost grows linearly with vectors x dims. HNSW (week 2) aims to stay fast as n grows.");
    return 0;
}
