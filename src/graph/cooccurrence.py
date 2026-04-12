import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass


@dataclass
class CooccurrenceData:
    C: sp.csr_matrix   # C[i,j] = # baskets containing both i and j
    N_items: np.ndarray  # N_items[i] = # baskets containing item i
    N: int              # total number of baskets
    n_products: int     # vocabulary size


def build_cooccurrence(
    train_baskets: list[set[int]],
    n_products: int,
) -> CooccurrenceData:
    """
    Build a sparse co-occurrence matrix from training baskets.

    Uses scipy.sparse CSR format. Only the upper triangle is populated
    during construction; the matrix is then symmetrized.

    Args:
        train_baskets: list of sets of product_idx values.
        n_products: total number of distinct products (vocabulary size).

    Returns:
        CooccurrenceData with C (CSR), N_items, N, n_products.
    """
    N = len(train_baskets)
    N_items = np.zeros(n_products, dtype=np.int32)

    rows, cols, data = [], [], []

    for basket in train_baskets:
        items = sorted(basket)
        for item in items:
            N_items[item] += 1
        for i_idx, i in enumerate(items):
            for j in items[i_idx + 1:]:
                rows.append(i)
                cols.append(j)
                data.append(1)

    C_upper = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_products, n_products),
        dtype=np.int32,
    )

    # Symmetrize: C = C_upper + C_upper.T
    C = C_upper + C_upper.T

    return CooccurrenceData(C=C, N_items=N_items, N=N, n_products=n_products)
