import pandas as pd


def load_transactions(
    csv_path: str,
    product_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Load and validate Dunnhumby transaction_data.csv.

    Keeps basket_id, product_id, day. If product_csv_path is provided,
    joins commodity_desc, sub_commodity_desc, department, brand from
    product.csv — useful for human-readable analysis and graph node labels.

    Returns:
        DataFrame with columns [basket_id, product_id, day] plus product
        metadata columns if product_csv_path was given.
    """
    df = pd.read_csv(
        csv_path,
        usecols=["BASKET_ID", "PRODUCT_ID", "DAY"],
        dtype={"BASKET_ID": int, "PRODUCT_ID": int, "DAY": int},
    )
    df = df.rename(columns={"BASKET_ID": "basket_id", "PRODUCT_ID": "product_id", "DAY": "day"})

    if product_csv_path is not None:
        products = pd.read_csv(
            product_csv_path,
            usecols=["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "DEPARTMENT", "BRAND"],
            dtype={"PRODUCT_ID": int},
        ).rename(columns={
            "PRODUCT_ID": "product_id",
            "COMMODITY_DESC": "commodity_desc",
            "SUB_COMMODITY_DESC": "sub_commodity_desc",
            "DEPARTMENT": "department",
            "BRAND": "brand",
        })
        df = df.merge(products, on="product_id", how="left")

    return df
