#!/usr/bin/env python3
"""Omnibenchmark-compatible random cluster assignment module."""
from __future__ import annotations
import argparse
import gzip
from pathlib import Path
import sys # this is important dont delete

import numpy as np
import pandas as pd


def assign_random_clusters(df: pd.DataFrame, num_clusters: int, seed: int | None = None) -> pd.DataFrame:
    """Return copy of df with a cluster_label column populated randomly."""
    if num_clusters < 1:
        raise ValueError("num_clusters must be at least 1")

    rng = np.random.default_rng(seed)
    labels = rng.integers(1, num_clusters + 1, size=len(df))
    result = df.copy()
    result["cluster_label"] = labels
    return result


def infer_separator(data_path: Path) -> str:
    """Guess delimiter from filename suffix (tsv/txt -> tab, else comma)."""
    suffixes = {suffix.lower() for suffix in data_path.suffixes}
    if ".tsv" in suffixes or ".txt" in suffixes:
        return "\t"
    return ","


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign random cluster labels to a gzipped matrix (Omnibenchmark module)."
    )
    parser.add_argument(
        "--data.true_labels",
        dest="data_true_labels",
        required=False,
        type=Path,
        help="Optional gzipped labels file (ignored, accepted for compatibility).",
    )
    parser.add_argument(
        "--data.matrix",
        dest="data_matrix",
        required=True,
        type=Path,
        help="Path to a gzipped CSV/TSV matrix file provided by Omnibenchmark.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory where Omnibenchmark expects the results.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name provided by Omnibenchmark (used in output file naming).",
    )
    parser.add_argument(
        "--num-clusters",
        dest="num_clusters",
        required=True,
        type=int,
        help="Number of clusters to sample from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional integer seed for reproducible results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    df = pd.read_csv(
        args.data_matrix,
        sep=infer_separator(args.data_matrix),
        header=None,
        compression="gzip",
    )
    clustered = assign_random_clusters(df, args.num_clusters, args.seed)
    labels = clustered["cluster_label"].to_numpy(dtype=int).reshape(-1, 1)
    header = np.array([[f"k={args.num_clusters}"]])
    output_matrix = np.vstack([header, labels.astype(str)])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.name}_ks_range.labels.gz"
    with gzip.open(output_file, "wt") as handle:
        np.savetxt(handle, output_matrix, fmt="%s", delimiter=",")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
