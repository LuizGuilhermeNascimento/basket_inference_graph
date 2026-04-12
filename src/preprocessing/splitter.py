import pandas as pd


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
