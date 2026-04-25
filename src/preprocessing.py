import pandas as pd
import numpy as np

_META_COLS = ["commodity_desc", "sub_commodity_desc", "department", "brand"]


def load_transactions(
    parquet_path: str,
    product_parquet_path: str | None = None,
) -> pd.DataFrame:
    """
    Load Dunnhumby transaction data from a Parquet file.

    Keeps basket_id, product_id, day. If product_parquet_path is provided,
    joins commodity_desc, sub_commodity_desc, department, brand —
    useful for human-readable analysis and graph node labels.

    Returns:
        DataFrame with columns [basket_id, product_id, day] plus product
        metadata columns if product_parquet_path was given.
    """
    df = pd.read_parquet(parquet_path, columns=["basket_id", "product_id", "day"])

    if product_parquet_path is not None:
        products = pd.read_parquet(
            product_parquet_path,
            columns=["product_id", "commodity_desc", "sub_commodity_desc", "department", "brand"],
        )
        df = df.merge(products, on="product_id", how="left")

    return df


def clean_transactions(
    df: pd.DataFrame,
    min_support: int,
) -> tuple[pd.DataFrame, dict[int, int], pd.DataFrame]:
    """
    Remove duplicates and rare products, then assign a canonical integer index.

    Steps:
    1. Drop duplicate (basket_id, product_id) rows.
    2. Remove products appearing in fewer than min_support baskets.
    3. Map surviving product_ids to a contiguous integer index [0, n_products).

    Returns:
        (cleaned_df, product_index, product_meta) where:
        - product_index maps original product_id → canonical integer index.
        - product_meta is a DataFrame indexed by product_idx with columns
          [product_id, commodity_desc, sub_commodity_desc, department, brand]
          (metadata columns are present only if loader joined product.csv).
    """
    df = df.drop_duplicates(subset=["basket_id", "product_id"])

    product_basket_counts = df.groupby("product_id")["basket_id"].nunique()
    frequent_products = product_basket_counts[product_basket_counts >= min_support].index
    df = df[df["product_id"].isin(frequent_products)].reset_index(drop=True)

    unique_products = sorted(df["product_id"].unique())
    product_index: dict[int, int] = {pid: idx for idx, pid in enumerate(unique_products)}
    df["product_idx"] = df["product_id"].map(product_index).astype(np.int32)

    # Build a one-row-per-product metadata lookup indexed by product_idx
    meta_cols = ["product_id"] + [c for c in _META_COLS if c in df.columns]
    product_meta = (
        df[meta_cols + ["product_idx"]]
        .drop_duplicates(subset=["product_id"])
        .set_index("product_idx")
        .sort_index()
    )

    return df, product_index, product_meta


def split_by_day(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[list[set[int]], list[set[int]]]:
    """
    Chronological train/test split based on the day of each basket.

    Baskets are assigned to train or test based on whether their day falls
    in the first train_fraction of days (by day number, not basket count).
    The split happens before graph construction to prevent data leakage.

    Returns:
        (train_baskets, test_baskets) — each a list of sets of product_idx values.
    """
    cutoff_day = df["day"].quantile(train_fraction)

    basket_day = df.groupby("basket_id")["day"].min()

    train_basket_ids = basket_day[basket_day <= cutoff_day].index
    test_basket_ids = basket_day[basket_day > cutoff_day].index

    def build_basket_sets(basket_ids: pd.Index) -> list[set[int]]:
        subset = df[df["basket_id"].isin(basket_ids)]
        return [
            set(group["product_idx"].tolist())
            for _, group in subset.groupby("basket_id")
        ]

    train_baskets = build_basket_sets(train_basket_ids)
    test_baskets = build_basket_sets(test_basket_ids)

    return train_baskets, test_baskets
