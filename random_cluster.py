#!/usr/bin/env python3
"""Omnibenchmark-compatible random cluster assignment module."""
from __future__ import annotations
import argparse
import gzip
from pathlib import Path
import sys # this is important dont delete

import numpy as np
import pandas as pd


def generate_k_grid(base_k: int) -> list[int]:
    """Return the [k-2, k-1, k, k+1, k+2] grid with each value >= 2."""
    if base_k < 1:
        raise ValueError("num_clusters must be at least 1")

    ks = [base_k - 2, base_k - 1, base_k, base_k + 1, base_k + 2]
    return [max(2, value) for value in ks]


def assign_random_clusters(num_rows: int, ks: list[int], seed: int | None = None) -> np.ndarray:
    """Return matrix of random cluster assignments per k in ks."""
    rng = np.random.default_rng(seed)
    columns = [rng.integers(1, k + 1, size=num_rows) for k in ks]
    return np.column_stack(columns)


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

    base_k = args.num_clusters
    if args.data_true_labels:
        labels_df = pd.read_csv(args.data_true_labels, header=None, compression="gzip")
        if labels_df.empty:
            raise SystemExit("Provided --data.true_labels is empty.")
        base_k = int(labels_df.iloc[:, 0].max())

    ks = generate_k_grid(base_k)
    label_matrix = assign_random_clusters(len(df), ks, args.seed)
    header = np.array([[f"k={k}" for k in ks]])
    output_matrix = np.vstack([header, label_matrix.astype(str)])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.name}_ks_range.labels.gz"
    with gzip.open(output_file, "wt") as handle:
        np.savetxt(handle, output_matrix, fmt="%s", delimiter=",")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
