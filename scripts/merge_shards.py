"""
Merge shard results into a single parquet file.

Usage:
    python3 scripts/merge_shards.py \
        --results-dir outputs/results \
        --output outputs/results/experiment_results_merged.parquet
"""

import argparse
import json
import os

import pandas as pd


def find_shards(results_dir: str) -> list[str]:
    entries = sorted(
        d for d in os.listdir(results_dir)
        if d.startswith("shard_") and os.path.isdir(os.path.join(results_dir, d))
    )
    if not entries:
        raise FileNotFoundError(f"No shard_* directories found in {results_dir}")
    return [os.path.join(results_dir, d) for d in entries]


def merge_experiment(shard_dirs: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(os.path.join(d, "experiment.parquet")) for d in shard_dirs]
    return pd.concat(frames, ignore_index=True)


def merge_hit_rates(shard_dirs: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(os.path.join(d, "product_hit_rates.parquet")) for d in shard_dirs]
    combined = pd.concat(frames, ignore_index=True)
    agg = (
        combined
        .groupby(["product_idx", "method", "k"], as_index=False)[["n_hidden", "n_hit"]]
        .sum()
    )
    agg["hit_rate"] = agg["n_hit"] / agg["n_hidden"].clip(lower=1)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge shard results into a single parquet")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--output", default="outputs/results/experiment_results_merged.parquet")
    args = parser.parse_args()

    shard_dirs = find_shards(args.results_dir)
    print(f"Found {len(shard_dirs)} shards: {[os.path.basename(d) for d in shard_dirs]}")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print("Merging experiment results ...")
    experiment_df = merge_experiment(shard_dirs)
    experiment_df.to_parquet(args.output, index=False)
    print(f"  {len(experiment_df):,} rows → {args.output}")

    hit_rates_path = os.path.join(output_dir, "product_hit_rates_merged.parquet")
    print("Merging product hit rates ...")
    hit_rates_df = merge_hit_rates(shard_dirs)
    hit_rates_df.to_parquet(hit_rates_path, index=False)
    print(f"  {len(hit_rates_df):,} rows → {hit_rates_path}")

    params_path = os.path.join(shard_dirs[0], "params.json")
    with open(params_path) as f:
        params = json.load(f)
    params["n_shards"] = len(shard_dirs)
    params.pop("shard_id", None)
    merged_params_path = os.path.join(output_dir, "params_merged.json")
    with open(merged_params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Params → {merged_params_path}")


if __name__ == "__main__":
    main()
