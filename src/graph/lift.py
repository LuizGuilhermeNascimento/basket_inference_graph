import numpy as np
import scipy.sparse as sp
from src.graph.cooccurrence import CooccurrenceData


def compute_lift(
    cooc: CooccurrenceData,
    min_lift: float = 1.0,
    min_cooccurrence: int = 5,
) -> sp.coo_matrix:
    """
    Compute lift for all item pairs with sufficient co-occurrence.

    lift(i→j) = (C[i,j] × N) / (N_items[i] × N_items[j])

    Pairs failing min_cooccurrence or min_lift thresholds are pruned.

    Args:
        cooc: output of build_cooccurrence.
        min_lift: keep edges with lift > min_lift (default 1.0).
        min_cooccurrence: discard pairs with fewer than this many co-occurrences.

    Returns:
        Sparse COO matrix of lift values for surviving (i, j) pairs.
    """
    C_coo = cooc.C.tocoo()

    mask = C_coo.data >= min_cooccurrence
    rows = C_coo.row[mask]
    cols = C_coo.col[mask]
    c_vals = C_coo.data[mask].astype(np.float64)

    lift_vals = (c_vals * cooc.N) / (cooc.N_items[rows] * cooc.N_items[cols])

    strong_mask = lift_vals > min_lift
    rows = rows[strong_mask]
    cols = cols[strong_mask]
    lift_vals = lift_vals[strong_mask]

    return sp.coo_matrix(
        (lift_vals, (rows, cols)),
        shape=(cooc.n_products, cooc.n_products),
    )
