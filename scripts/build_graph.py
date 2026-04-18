"""
End-to-end runner for Phases 1 and 2.

Usage:
    python scripts/build_graph.py \
        --data "data/raw/transaction_data.csv" \
        --products "data/raw/product.csv" \
        --output outputs/graphs

Produces:
    outputs/graphs/association_graph.graphml
    outputs/graphs/association_graph.gexf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.loader import load_transactions
from src.preprocessing.cleaner import clean_transactions
from src.preprocessing.splitter import split_by_day
from src.graph.cooccurrence import build_cooccurrence
from src.graph.lift import compute_lift
from src.graph.builder import build_graph, save_graph, load_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build association graph from Dunnhumby data")
    parser.add_argument("--data", default="data/raw/transaction_data.csv", help="Path to transaction_data.csv")
    parser.add_argument("--products", default="data/raw/product.csv", help="Path to product.csv (enables node labels)")
    parser.add_argument("--output", default="outputs/graphs", help="Output directory for graph files")
    parser.add_argument(
        "--processed-output",
        default="data/processed/transactions_processed.parquet",
        help="Path to write processed transactions as parquet",
    )
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-lift", type=float, default=1.0)
    parser.add_argument("--min-cooccurrence", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    print("Phase 1 — Preprocessing")
    print(f"  Loading {args.data} ...")
    df = load_transactions(args.data, product_csv_path=args.products)
    print(f"  Loaded {len(df):,} rows | {df['basket_id'].nunique():,} baskets | {df['product_id'].nunique():,} products")
    if args.products:
        print(f"  Product metadata joined from {args.products}")

    df, product_index, product_meta = clean_transactions(df, min_support=args.min_support)
    n_products = len(product_index)
    print(f"  After cleaning: {len(df):,} rows | {df['basket_id'].nunique():,} baskets | {n_products:,} products")

    processed_dir = os.path.dirname(args.processed_output)
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
    df.to_parquet(args.processed_output, index=False)
    print(f"  Saved processed parquet: {args.processed_output}")

    train_baskets, test_baskets = split_by_day(df, train_fraction=args.train_fraction)
    print(f"  Train baskets: {len(train_baskets):,} | Test baskets: {len(test_baskets):,}")

    print("\nPhase 2 — Graph construction")
    print("  Building co-occurrence matrix ...")
    cooc = build_cooccurrence(train_baskets, n_products)
    print(f"  Co-occurrence matrix: {cooc.C.nnz:,} nonzero entries | N={cooc.N:,} baskets")

    print(f"  Computing lift (min_lift={args.min_lift}, min_cooccurrence={args.min_cooccurrence}) ...")
    lift_matrix = compute_lift(cooc, min_lift=args.min_lift, min_cooccurrence=args.min_cooccurrence)
    print(f"  Lift matrix: {lift_matrix.nnz:,} surviving pairs")

    print("  Assembling DiGraph ...")
    G = build_graph(lift_matrix, product_meta=product_meta if args.products else None)
    print(f"  Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} directed edges")

    print(f"  Saving to {args.output} ...")
    save_graph(G, args.output)

    # Verify round-trip weight preservation
    G_check = load_graph(args.output, fmt="graphml")
    sample = list(G.edges(data=True))[0]
    i, j, attrs = sample
    loaded_weight = G_check[i][j]["weight"]
    assert abs(loaded_weight - attrs["weight"]) < 1e-9, (
        f"Weight mismatch after serialization: {attrs['weight']} vs {loaded_weight}"
    )
    print("  Round-trip weight check passed.")

    print(f"\nDone. Graph written to:")
    print(f"  {args.output}/association_graph.graphml")
    print(f"  {args.output}/association_graph.gexf")


if __name__ == "__main__":
    main()
