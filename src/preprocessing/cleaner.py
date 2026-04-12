import pandas as pd
import numpy as np

_META_COLS = ["commodity_desc", "sub_commodity_desc", "department", "brand"]


def clean_transactions(
    df: pd.DataFrame,
    min_support: int = 50,
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
