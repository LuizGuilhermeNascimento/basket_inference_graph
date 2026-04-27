"""
Convert all CSV files in input_path to Parquet and save in output_path.

Usage:
    python scripts/csv_to_parquet.py <input_path> <output_path>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(input_path: Path, output_path: Path) -> None:
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        out_file = output_path / csv_file.with_suffix(".parquet").name
        df.to_parquet(out_file, index=False)
        print(f"{csv_file.name} → {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV files to Parquet.")
    parser.add_argument("input_path", type=Path, help="Directory containing CSV files")
    parser.add_argument("output_path", type=Path, help="Directory to save Parquet files")
    args = parser.parse_args()

    if not args.input_path.is_dir():
        print(f"Error: {args.input_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    convert_csv_to_parquet(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
