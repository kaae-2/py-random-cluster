#!/usr/bin/env python3
"""Command line random cluster assignment using pandas."""
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
        "matrix", type=Path, help="Path to a CSV/TSV file (optionally gzipped) containing the matrix"
    )
    parser.add_argument("num_clusters", type=int, help="Number of clusters to sample from")
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
        "--output",
        type=Path,
        help="Optional path to write the matrix with cluster labels. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--compression",
        default="infer",
        help=(
            "Compression type understood by pandas.read_csv (e.g. 'gzip'). "
            "Default: infer from filename"
        ),
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

    df = pd.read_csv(args.matrix, sep=args.sep, header=header, compression=args.compression)
    clustered = assign_random_clusters(df, args.num_clusters, args.seed)

    if args.output:
        clustered.to_csv(args.output, index=False)
    else:
        clustered.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
