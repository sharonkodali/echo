# Echo

[![ci](https://github.com/sharonkodali/echo/actions/workflows/ci.yml/badge.svg)](https://github.com/sharonkodali/echo/actions/workflows/ci.yml)

**A vector search engine in C++: find the songs that echo the one you picked.**

Every song becomes a point in space, built from its audio features
(danceability, energy, mood, tempo...). Similar songs are points close to each
other, so "find songs like this one" becomes "find the nearest points".

Echo is being built in stages:

- [x] **Week 1: exact search.** Brute-force nearest neighbors, the answer key everything else is graded against
- [ ] **Week 2: HNSW.** A layered graph index that finds almost the same answers while checking a tiny fraction of the points
- [ ] **Week 3: benchmarks.** Recall vs. speed curves, parameter sweeps, SIFT1M, comparison with hnswlib
- [ ] **Week 4: live demo.** Compiled to WebAssembly, powering a "find similar songs" box on [VibeCanvas](https://github.com/sharonkodali/vibecanvas)

## Try it

```bash
python3 scripts/prepare_songs.py          # downloads 28,351 Spotify tracks from VibeCanvas -> data/songs.tsv
cmake -S . -B build && cmake --build build -j
./build/echo_query data/songs.tsv "blinding lights"
```

```
Songs like: Blinding Lights - The Weeknd (pop)

 1. Part-Time Lover                               Dabin                     edm    d=0.0079
 2. POP/STARS                                     K/DA                      pop    d=0.0129
 3. Criminal                                      Framing Hanley            rock   d=0.0155
 ...
Searched 28351 songs (9 features each) in 0.152 ms (exact brute force)
```

Run the tests with `ctest --test-dir build --output-on-failure`.

## How it works (week 1)

| Piece | File | Idea |
|---|---|---|
| Song vectors | [`scripts/prepare_songs.py`](scripts/prepare_songs.py) | 9 audio features per song, **standardized** (mean 0, std 1). Otherwise tempo (60–200) and loudness (−60–0 dB) would drown out danceability and energy (0–1). |
| Storage | [`include/echo/dataset.h`](include/echo/dataset.h) | All vectors in **one contiguous array**, not a vector of vectors, so the CPU streams through memory instead of chasing pointers. |
| Distance | [`include/echo/distance.h`](include/echo/distance.h) | Squared Euclidean (no square root: it doesn't change the ranking) and cosine (direction only, via normalized dot product). |
| Exact search | [`include/echo/brute_force.h`](include/echo/brute_force.h) | Check every song, keeping the best *k* in a **max-heap**: O(n log k) instead of sorting everything. Ties broken by index so results are deterministic. |
| Recall | same file | Fraction of the true nearest neighbors an approximate index found. HNSW will be scored with this. |

## Why brute force isn't enough

`./build/bench_bruteforce data/songs.tsv` on an Apple M5 laptop:

| Data | Vectors | Dims | ms / query |
|---|---:|---:|---:|
| Songs (real) | 28,351 | 9 | 0.37 |
| Random | 100,000 | 16 | 0.40 |
| Random | 1,000,000 | 16 | 2.99 |
| Random | 100,000 | 128 | 4.03 |

Exact search costs time proportional to *vectors × dimensions* on every
query. For 28K songs that's instant, so the song demo alone wouldn't justify
HNSW. At millions of vectors or hundreds of dimensions (real embeddings) it
becomes the bottleneck, and that's where week 3 will show where HNSW pays off.

## Notes on the data

- 28,351 tracks from the Kaggle "30000 Spotify Songs" dataset, as published by VibeCanvas.
- Some songs appear as several releases with different IDs (e.g. "Bohemian Rhapsody - 2011 Mix" and "Remastered 2011"). They have identical audio features, so they show up as each other's nearest neighbor at distance 0.

## Layout

```
include/echo/   public headers (distance, dataset, brute_force)
src/            implementations
tools/          echo_query (search from the terminal), bench_bruteforce
tests/          GoogleTest suite + small fixture files
scripts/        prepare_songs.py (download + standardize the song data)
```

CI builds and tests on Linux and macOS, runs the suite under AddressSanitizer
and UndefinedBehaviorSanitizer, and runs an end-to-end query on the real data.

## License

MIT
