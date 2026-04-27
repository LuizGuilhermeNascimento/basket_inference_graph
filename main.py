"""
Usage:
    python3 main.py \
        --data "data/raw/transaction_data.parquet" \
        --products "data/raw/product.parquet" \
        --output outputs/graphs

Produces:
    outputs/graphs/association_graph.graphml
    outputs/graphs/association_graph.gexf
"""

import argparse
import os

from src.preprocessing import load_transactions, clean_transactions, split_by_day
from src.graph_builder import build_cooccurrence, compute_confidence, build_graph, save_graph, load_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build association graph from Dunnhumby data")
    parser.add_argument("--data", default="data/raw/transaction_data.parquet", help="Path to transaction_data.parquet")
    parser.add_argument("--products", default=None, help="Path to product.parquet (enables node labels)")
    parser.add_argument("--output", default="outputs/graphs", help="Output directory for graph files")
    parser.add_argument(
        "--processed-output",
        default="data/processed/transactions_processed.parquet",
        help="Path to write processed transactions as parquet",
    )
    parser.add_argument("--train-output", default="data/processed/train.parquet")
    parser.add_argument("--test-output", default="data/processed/test.parquet")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-cooccurrence", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    print("Phase 1 — Preprocessing")
    print(f"  Loading {args.data} ...")
    df = load_transactions(args.data, product_parquet_path=args.products)
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

    train_baskets, test_baskets, cutoff_day = split_by_day(df, train_fraction=args.train_fraction)
    print(f"  Cutoff day: {cutoff_day} | Train baskets: {len(train_baskets):,} | Test baskets: {len(test_baskets):,}")

    basket_day = df.groupby("basket_id")["day"].min()
    train_ids = basket_day[basket_day <= cutoff_day].index
    train_df = df[df["basket_id"].isin(train_ids)]
    test_df = df[~df["basket_id"].isin(train_ids)]
    train_df.to_parquet(args.train_output, index=False)
    test_df.to_parquet(args.test_output, index=False)
    print(f"  Saved train parquet: {args.train_output} ({len(train_df):,} rows)")
    print(f"  Saved test parquet:  {args.test_output} ({len(test_df):,} rows)")

    print("\nPhase 2 — Graph construction")
    print("  Building co-occurrence matrix ...")
    cooc = build_cooccurrence(train_baskets, n_products)
    print(f"  Co-occurrence matrix: {cooc.C.nnz:,} nonzero entries | N={cooc.N:,} baskets")

    print(f"  Computing confidence (min_cooccurrence={args.min_cooccurrence}) ...")
    weight_matrix = compute_confidence(cooc, min_cooccurrence=args.min_cooccurrence)
    print(f"  Directed edges: {weight_matrix.nnz:,}")

    print("  Assembling DiGraph ...")
    G = build_graph(weight_matrix, product_meta=product_meta if args.products else None)
    print(f"  Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} directed edges")

    print(f"  Saving to {args.output} ...")
    save_graph(G, args.output)

    print(f"\nDone. Graph written to:")
    print(f"  {args.output}/association_graph.graphml")
    print(f"  {args.output}/association_graph.gexf")


if __name__ == "__main__":
    main()
