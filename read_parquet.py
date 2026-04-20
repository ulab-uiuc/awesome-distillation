#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import random
import pandas as pd


def pretty_print_value(v, max_str_len=3000):
    """Pretty print a cell value."""
    if isinstance(v, (dict, list)):
        try:
            s = json.dumps(v, ensure_ascii=False, indent=2)
        except Exception:
            s = str(v)
    else:
        s = str(v)

    if len(s) > max_str_len:
        s = s[:max_str_len] + "\n... [truncated]"
    return s


def show_sample(df, idx, max_str_len=3000):
    print("\n" + "=" * 100)
    print(f"Sample index: {idx}")
    print("=" * 100)
    row = df.iloc[idx]
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(pretty_print_value(row[col], max_str_len=max_str_len))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="/root/math/data/sft_qwen3_1p7b_generated_openthoughts_math.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random samples to show",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-str-len",
        type=int,
        default=3000,
        help="Maximum printed length per field",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Reading: {file_path}")
    df = pd.read_parquet(file_path)

    print("\n=== Basic Info ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")

    n = len(df)
    if n == 0:
        print("\nDataset is empty.")
        return

    k = min(args.num_samples, n)
    random.seed(args.seed)
    indices = random.sample(range(n), k)

    print(f"\nRandomly sampled {k} indices with seed={args.seed}: {indices}")

    for idx in indices:
        show_sample(df, idx, max_str_len=args.max_str_len)


if __name__ == "__main__":
    main()