#!/usr/bin/env python3
"""Omnibenchmark-compatible random cluster assignment module."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign random cluster labels (1..num_clusters) to each row of a matrix "
            "loaded via pandas."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a gzipped CSV/TSV file containing the matrix",
    )
    parser.add_argument(
        "--num_clusters",
        required=True,
        type=int,
        help="Number of clusters to sample from",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Column delimiter understood by pandas.read_csv (default: ',')",
    )
    parser.add_argument(
        "--header",
        type=str,
        default="0",
        help=(
            "Header row index understood by pandas.read_csv (use 'None' for no header). "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory where Omnibenchmark expects the results",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name provided by Omnibenchmark (used in output file naming)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional integer seed for reproducible results",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    header: int | None
    if args.header.lower() == "none":
        header = None
    else:
        try:
            header = int(args.header)
        except ValueError:  # pragma: no cover - arg parsing limits testability
            raise SystemExit("--header must be an integer or 'None'")

    df = pd.read_csv(args.input, sep=args.sep, header=header, compression="gzip")
    clustered = assign_random_clusters(df, args.num_clusters, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.name}_random_clusters.csv.gz"
    clustered.to_csv(output_file, index=False, compression="gzip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
