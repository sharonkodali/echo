// echo_query: "find songs like this one" from the terminal.
//
//   ./build/echo_query data/songs.tsv "blinding lights"
//   ./build/echo_query data/songs.tsv "blinding lights" --k 5 --metric l2
//
// Week 1 uses exact brute-force search. Week 2 swaps in HNSW behind the
// same interface, and this tool can compare the two.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "echo/brute_force.h"
#include "echo/dataset.h"

namespace {

// Columns produced by scripts/prepare_songs.py.
constexpr std::size_t kNumMetaCols = 4;  // id, name, artist, genre
constexpr std::size_t kName = 1, kArtist = 2, kGenre = 3;

int usage() {
    std::cerr << "usage: echo_query <songs.tsv> <song name> [--k N] [--metric cosine|l2]\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string path = argv[1];
    const std::string query_name = argv[2];
    std::size_t k = 10;
    echo::Metric metric = echo::Metric::Cosine;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--k" && i + 1 < argc) {
            k = std::strtoul(argv[++i], nullptr, 10);
        } else if (arg == "--metric" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "l2") metric = echo::Metric::L2;
            else if (m == "cosine") metric = echo::Metric::Cosine;
            else return usage();
        } else {
            return usage();
        }
    }

    try {
        const auto data = echo::Dataset::load_tsv(path, kNumMetaCols);
        const auto matches = data.find(kName, query_name);
        if (matches.empty()) {
            std::cerr << "no song matching \"" << query_name << "\"\n";
            return 1;
        }
        const std::uint32_t song = matches.front();
        const auto& m = data.meta(song);
        std::cout << "Songs like: " << m[kName] << " - " << m[kArtist] << " (" << m[kGenre] << ")\n";
        if (matches.size() > 1) {
            std::cout << "(" << matches.size() - 1 << " other matches; using the first)\n";
        }

        const echo::BruteForceIndex index(data, metric);
        const auto start = std::chrono::steady_clock::now();
        const auto results = index.search(data.row(song), k, song);
        const auto elapsed = std::chrono::steady_clock::now() - start;

        std::cout << "\n";
        for (std::size_t r = 0; r < results.size(); ++r) {
            const auto& meta = data.meta(results[r].index);
            std::printf("%2zu. %-45.45s %-25.25s %-6s d=%.4f\n", r + 1, meta[kName].c_str(),
                        meta[kArtist].c_str(), meta[kGenre].c_str(), results[r].distance);
        }
        std::printf("\nSearched %zu songs (%zu features each) in %.3f ms (exact brute force)\n",
                    index.size(), index.dim(),
                    std::chrono::duration<double, std::milli>(elapsed).count());
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
