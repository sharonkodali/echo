#!/usr/bin/env python3
"""Download the VibeCanvas Spotify tracks and turn each song into a vector.

    python3 scripts/prepare_songs.py            # writes data/songs.tsv

Each song becomes 9 numbers (its audio features). They are standardized
(z-scored): every feature is shifted to mean 0 and scaled to standard
deviation 1. Without this, tempo (~60-200) and loudness (~-60-0 dB) would
swamp danceability or energy (0-1) in every distance calculation, and
"similar" would mostly mean "similar tempo".

Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from pathlib import Path

SOURCE = "https://sharonkodali.github.io/vibecanvas/site-data/tracks_{}.json"
NUM_FILES = 4

# JSON key -> column name. These are Spotify's audio features.
FEATURES = {
    "d": "danceability",
    "e": "energy",
    "v": "valence",
    "t": "tempo",
    "ac": "acousticness",
    "ins": "instrumentalness",
    "sp": "speechiness",
    "li": "liveness",
    "lo": "loudness",
}


def clean(text: str) -> str:
    """Tabs or newlines in a song title would break the TSV format."""
    return " ".join(str(text).split())


def load_tracks(local_dir: Path | None) -> list[dict]:
    tracks = []
    for i in range(NUM_FILES):
        if local_dir:
            raw = (local_dir / f"tracks_{i}.json").read_text()
        else:
            with urllib.request.urlopen(SOURCE.format(i), timeout=30) as resp:
                raw = resp.read().decode()
        tracks.extend(json.loads(raw))
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/songs.tsv"))
    parser.add_argument("--local", type=Path, help="read tracks_*.json from this folder instead of downloading")
    args = parser.parse_args()

    tracks = load_tracks(args.local)

    # The same song can appear in several playlists: keep the first copy, and
    # skip any track missing a feature.
    seen, songs = set(), []
    for t in tracks:
        if t["id"] in seen or any(t.get(k) is None for k in FEATURES):
            continue
        seen.add(t["id"])
        songs.append(t)

    stats = {}
    for key in FEATURES:
        values = [float(s[key]) for s in songs]
        stats[key] = (statistics.fmean(values), statistics.pstdev(values) or 1.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("\t".join(["id", "name", "artist", "genre", *FEATURES.values()]) + "\n")
        for s in songs:
            zs = [(float(s[k]) - mean) / std for k, (mean, std) in stats.items()]
            meta = [clean(s["id"]), clean(s["n"]), clean(s["a"]), clean(s["g"])]
            f.write("\t".join(meta + [f"{z:.6f}" for z in zs]) + "\n")

    print(f"{len(tracks)} tracks downloaded, {len(songs)} unique songs -> {args.out} "
          f"({len(FEATURES)} standardized features each)")


if __name__ == "__main__":
    main()
