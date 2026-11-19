"""Fetch the benchmark parquet tables into data/.

The Hugging Face dataset ``lucianfialho/synthetic-attribution-benchmark``
(revision ``main``, commit 293a1054afa9ab1abefb21ab2ff14914d7bb52f0) stores
the tables as Git LFS files at the repository root. There is no ``data/``
directory on the hub.

    journeys.parquet      5,472,065 bytes
    touchpoints.parquet  12,772,646 bytes
    ground_truth.json        1,136 bytes   (already committed here)

``datasets.load_dataset`` is the wrong loader for these tables. The hub
tags the repo ``format:imagefolder`` because it also contains PNGs, so the
datasets viewer does not expose the parquet files. This script resolves the
two LFS files with ``huggingface_hub`` and writes them where
``run_benchmark.py`` reads them: ``data/journeys.parquet`` and
``data/touchpoints.parquet``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "lucianfialho/synthetic-attribution-benchmark"
# main as checked on 2026-09-17. Pinning the commit keeps the tables aligned
# with the committed data/ground_truth.json.
REVISION = "293a1054afa9ab1abefb21ab2ff14914d7bb52f0"
FILES = ("journeys.parquet", "touchpoints.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.dest,
        ))
        print(f"{name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
