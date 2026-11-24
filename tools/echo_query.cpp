// echo_query: "find songs like this one" from the terminal.
//
//   ./build/echo_query data/songs.tsv "blinding lights"
//   ./build/echo_query data/songs.tsv "blinding lights" --k 5 --metric l2
//   ./build/echo_query data/songs.tsv "blinding lights" --index hnsw --ef 32
//
// --index brute (default) is exact. --index hnsw searches the graph and
// also reports its recall against the exact answer.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "echo/brute_force.h"
#include "echo/dataset.h"
#include "echo/hnsw.h"

namespace {

// Columns produced by scripts/prepare_songs.py.
constexpr std::size_t kNumMetaCols = 4;  // id, name, artist, genre
constexpr std::size_t kName = 1, kArtist = 2, kGenre = 3;

int usage() {
    std::cerr << "usage: echo_query <songs.tsv> <song name> [--k N] [--metric cosine|l2]\n"
                 "                  [--index brute|hnsw] [--ef N]\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string path = argv[1];
    const std::string query_name = argv[2];
    std::size_t k = 10;
    echo::Metric metric = echo::Metric::Cosine;
    bool use_hnsw = false;
    std::size_t ef = 0;  // 0 = index default

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--k" && i + 1 < argc) {
            k = std::strtoul(argv[++i], nullptr, 10);
        } else if (arg == "--metric" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "l2") metric = echo::Metric::L2;
            else if (m == "cosine") metric = echo::Metric::Cosine;
            else return usage();
        } else if (arg == "--index" && i + 1 < argc) {
            const std::string idx = argv[++i];
            if (idx == "hnsw") use_hnsw = true;
            else if (idx != "brute") return usage();
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = std::strtoul(argv[++i], nullptr, 10);
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

        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };

        const echo::BruteForceIndex exact(data, metric);
        auto start = Clock::now();
        const auto truth = exact.search(data.row(song), k, song);
        const double exact_ms = ms(Clock::now() - start);

        std::vector<echo::Neighbor> results = truth;
        double hnsw_ms = 0, build_ms = 0;
        if (use_hnsw) {
            start = Clock::now();
            const echo::HnswIndex hnsw(data, metric);
            build_ms = ms(Clock::now() - start);
            start = Clock::now();
            results = hnsw.search(data.row(song), k, song, ef);
            hnsw_ms = ms(Clock::now() - start);
        }

        std::cout << "\n";
        for (std::size_t r = 0; r < results.size(); ++r) {
            const auto& meta = data.meta(results[r].index);
            std::printf("%2zu. %-45.45s %-25.25s %-6s d=%.4f\n", r + 1, meta[kName].c_str(),
                        meta[kArtist].c_str(), meta[kGenre].c_str(), results[r].distance);
        }
        std::printf("\nSearched %zu songs (%zu features each)\n", exact.size(), exact.dim());
        std::printf("  exact brute force: %.3f ms\n", exact_ms);
        if (use_hnsw) {
            std::printf("  hnsw:              %.3f ms, recall@%zu = %.2f (graph built in %.0f ms)\n",
                        hnsw_ms, k, echo::recall(results, truth), build_ms);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
