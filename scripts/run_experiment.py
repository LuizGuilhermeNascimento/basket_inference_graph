"""
Run the basket completion experiment.

Usage:
python3 scripts/run_experiment.py --include-all --k-values 1 2 5 \
  --n-repetitions 10 --min-basket-size 2 --n-workers 5 \
  --checkpoint-every 200 \
  --max-basket-size 5 \
  --output outputs/results/experiment_results_full.parquet

Loads the association graph + item counts, builds three recommenders
(Popularity, LocalAgg, PPR), evaluates on test baskets, and writes results.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_builder import load_graph
from src.recommenders import PopularityRecommender, LocalAggRecommender, PPRRecommender
from src.evaluation import run_basket_completion_experiment

RANDOM_SEED = 42


def _load_item_counts(graph_dir: str, train_parquet: str) -> np.ndarray:
    path = os.path.join(graph_dir, "item_counts.npy")
    if os.path.exists(path):
        print(f"  Loaded item_counts.npy from {path}")
        return np.load(path)

    print(f"  item_counts.npy not found — computing from {train_parquet}")
    train = pd.read_parquet(train_parquet, columns=["basket_id", "product_idx"])
    n_products = int(train["product_idx"].max()) + 1
    counts = np.zeros(n_products, dtype=np.int32)
    for products in train.groupby("basket_id")["product_idx"].unique():
        for p in products:
            counts[p] += 1
    return counts


def _graph_to_csr(G, n_products: int) -> sp.csr_matrix:
    if G.number_of_edges() == 0:
        return sp.csr_matrix((n_products, n_products))
    rows, cols, weights = zip(*((u, v, d["weight"]) for u, v, d in G.edges(data=True)))
    return sp.csr_matrix((list(weights), (list(rows), list(cols))), shape=(n_products, n_products))


def _row_normalize(W: sp.csr_matrix) -> sp.csr_matrix:
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    return sp.diags(1.0 / row_sums).dot(W).tocsr()


def _load_test_baskets(path: str) -> list[set[int]]:
    df = pd.read_parquet(path, columns=["basket_id", "product_idx"])
    return [
        set(group["product_idx"].tolist())
        for _, group in df.groupby("basket_id")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basket completion experiment")
    parser.add_argument("--graph-dir", default="outputs/graphs")
    parser.add_argument("--test-data", default="data/processed/test.parquet")
    parser.add_argument("--train-data", default="data/processed/train.parquet",
                        help="Used to recompute item counts if item_counts.npy is missing")
    parser.add_argument("--output", default="outputs/results/experiment_results.parquet")
    parser.add_argument(
        "--n-obs", nargs="+", type=str, default=["0.25", "0.5", "0.75"],
        help="n_obs specs: fractions 0<p<1 (e.g. 0.25 0.5 0.75), absolute ints (e.g. 1 2 3), "
             "or named specs 'all'/'half'. Fractions are resolved per basket as ceil(p*|T|).",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--n-repetitions", type=int, default=10)
    parser.add_argument("--min-basket-size", type=int, default=5)
    parser.add_argument("--max-basket-size", type=int, default=None,
                        help="Skip baskets larger than this (None = no limit)")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--min-cooccurrence", type=int, default=2,
                        help="Must match the value used during graph construction (logged only)")
    parser.add_argument("--n-shards", type=int, default=1,
                        help="Split test set into this many shards and run one shard at a time")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="0-indexed shard to process (must be < --n-shards)")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="Save partial results every N baskets evaluated (0 = only at the end)")
    parser.add_argument("--n-workers", type=int, default=3,
                        help="Threads for parallel basket processing (scipy releases GIL during matmul)")
    args = parser.parse_args()

    if not (0 <= args.shard_id < args.n_shards):
        raise ValueError(f"--shard-id must be in [0, {args.n_shards - 1}]")

    # Derive output paths — inject shard subdirectory when sharding is active
    base_output = os.path.abspath(args.output)
    if args.n_shards > 1:
        results_path = os.path.join(
            os.path.dirname(base_output),
            f"shard_{args.shard_id:02d}",
            os.path.basename(base_output),
        )
    else:
        results_path = base_output
    output_dir = os.path.dirname(results_path)
    hit_rates_path = os.path.join(output_dir, "product_hit_rates.parquet")

    print("=" * 60)
    print("Basket Completion Experiment")
    print("=" * 60)
    print(f"Config: seed={args.seed}, min_cooccurrence={args.min_cooccurrence}, "
          f"min_basket_size={args.min_basket_size}")
    if args.n_shards > 1:
        print(f"Shard: {args.shard_id + 1}/{args.n_shards}")

    print("\n[1/5] Loading graph ...")
    G = load_graph(args.graph_dir)
    graph_products = set(G.nodes())
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges")

    print("\n[2/5] Loading item counts ...")
    item_counts = _load_item_counts(args.graph_dir, args.train_data)
    n_products = len(item_counts)
    print(f"  n_products={n_products}")

    print("\n[3/5] Building confidence matrix ...")
    W = _graph_to_csr(G, n_products)
    W_norm = _row_normalize(W)
    print(f"  Confidence matrix: {W.nnz} nonzero entries")

    print("\n[4/5] Loading test baskets ...")
    all_baskets = _load_test_baskets(args.test_data)
    print(f"  {len(all_baskets)} test baskets total")

    # Shard: replace out-of-shard baskets with empty sets so basket_ids stay stable.
    # The empty sets are filtered out by min_basket_size inside the experiment.
    if args.n_shards > 1:
        test_baskets = [
            basket if bid % args.n_shards == args.shard_id else set()
            for bid, basket in enumerate(all_baskets)
        ]
        n_shard = sum(1 for b in test_baskets if b)
        print(f"  Shard {args.shard_id}: {n_shard} baskets (basket_ids preserved)")
    else:
        test_baskets = all_baskets

    print("\n[5/5] Running experiment ...")
    n_obs_values = args.n_obs

    recommenders = {
        "popularity": PopularityRecommender(item_counts),
        "local_agg": LocalAggRecommender(W),
        "ppr": PPRRecommender(W_norm, alpha=args.alpha),
    }

    print(f"  n_obs={n_obs_values}, k_values={args.k_values}, "
          f"n_reps={args.n_repetitions}, alpha={args.alpha}, "
          f"n_workers={args.n_workers}, checkpoint_every={args.checkpoint_every}, "
          f"max_basket_size={args.max_basket_size}")

    os.makedirs(output_dir, exist_ok=True)

    params = {
        "seed": args.seed,
        "min_cooccurrence": args.min_cooccurrence,
        "min_basket_size": args.min_basket_size,
        "max_basket_size": args.max_basket_size,
        "n_obs": n_obs_values,
        "k_values": args.k_values,
        "n_repetitions": args.n_repetitions,
        "alpha": args.alpha,
        "n_shards": args.n_shards,
        "shard_id": args.shard_id,
        "n_workers": args.n_workers,
    }
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Params saved → {params_path}")

    def checkpoint_fn(partial_results: pd.DataFrame, partial_hit_stats: pd.DataFrame) -> None:
        partial_results.to_parquet(results_path, index=False)
        hit_rates = partial_hit_stats.copy()
        hit_rates["hit_rate"] = hit_rates["n_hit"] / hit_rates["n_hidden"].clip(lower=1)
        hit_rates.to_parquet(hit_rates_path, index=False)
        print(f"  [checkpoint] {len(partial_results):,} rows saved → {results_path}")

    results_df, hit_stats_df = run_basket_completion_experiment(
        test_baskets=test_baskets,
        graph_products=graph_products,
        recommenders=recommenders,
        n_obs_values=n_obs_values,
        k_values=args.k_values,
        n_repetitions=args.n_repetitions,
        min_basket_size=args.min_basket_size,
        max_basket_size=args.max_basket_size,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_fn=checkpoint_fn,
        n_workers=args.n_workers,
    )

    print(f"  Done — {len(results_df):,} result rows")

    # Final save (overwrites last checkpoint with complete results)
    checkpoint_fn(results_df, hit_stats_df)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {hit_rates_path}")


if __name__ == "__main__":
    main()
