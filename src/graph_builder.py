import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------

@dataclass
class CooccurrenceData:
    C: sp.csr_matrix    # C[i,j] = # baskets containing both i and j
    N_items: np.ndarray  # N_items[i] = # baskets containing item i
    N: int               # total number of baskets
    n_products: int      # vocabulary size


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


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backbone (Disparity Filter)
# ---------------------------------------------------------------------------

def disparity_filter(
    lift_matrix: sp.coo_matrix,
    alpha: float = 0.05,
    bidirectional: bool = True,
) -> sp.coo_matrix:
    """
    Extract the backbone of a weighted directed graph via the Disparity Filter
    (Serrano et al., 2009).

    For each source node i with out-degree k_i and strength s_i = Σ_j w_ij:
        p_ij  = w_ij / s_i
        α_ij  = (1 - p_ij)^(k_i - 1)

    Keeps edge (i→j) if α_ij < alpha. With bidirectional=True, keeps the edge
    if it passes the test from either the src or dst perspective.

    Args:
        lift_matrix: sparse COO matrix of lift values (shape n×n).
        alpha: significance threshold (default 0.05).
        bidirectional: keep edge if significant from src OR dst perspective.

    Returns:
        Filtered sparse COO matrix with the same shape as the input.
    """
    rows = lift_matrix.row.astype(np.int64)
    cols = lift_matrix.col.astype(np.int64)
    weights = lift_matrix.data.astype(np.float64)
    n = lift_matrix.shape[0]

    degree_src, strength_src = _node_stats(rows, weights, n)
    alpha_src = _significance(weights, rows, degree_src, strength_src)

    if not bidirectional:
        mask = alpha_src < alpha
    else:
        degree_dst, strength_dst = _node_stats(cols, weights, n)
        alpha_dst = _significance(weights, cols, degree_dst, strength_dst)
        mask = (alpha_src < alpha) | (alpha_dst < alpha)

    return sp.coo_matrix(
        (weights[mask], (rows[mask].astype(np.int32), cols[mask].astype(np.int32))),
        shape=lift_matrix.shape,
    )


def _node_stats(
    node_indices: np.ndarray,
    weights: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    degree = np.bincount(node_indices, minlength=n).astype(np.float64)
    strength = np.bincount(node_indices, weights=weights, minlength=n)
    return degree, strength


def _significance(
    weights: np.ndarray,
    src_indices: np.ndarray,
    degree: np.ndarray,
    strength: np.ndarray,
) -> np.ndarray:
    k = degree[src_indices]
    s = strength[src_indices]
    p = weights / s
    # Nodes with degree=1 have k-1=0, so alpha=1 — never significant, correctly filtered out
    return (1 - p) ** (k - 1)


# ---------------------------------------------------------------------------
# Graph assembly and I/O
# ---------------------------------------------------------------------------

def build_graph(
    lift_matrix: sp.coo_matrix,
    product_meta: pd.DataFrame | None = None,
) -> nx.DiGraph:
    """
    Assemble a weighted directed graph from the lift COO matrix.

    Each nonzero entry (i, j) in lift_matrix produces two directed edges:
    i→j and j→i, both with weight = lift(i,j). Lift is symmetric, so
    both directions get the same weight. Keeping both directions allows
    the scorer to sum over observed items i ∈ S via G[i][j]['weight'].

    If product_meta is provided (DataFrame indexed by product_idx with columns
    like commodity_desc, department, brand), those fields are attached as node
    attributes — useful for Gephi visualisation and analysis notebooks.

    Args:
        lift_matrix: sparse COO matrix with lift values.
        product_meta: optional metadata DataFrame from clean_transactions.

    Returns:
        nx.DiGraph with integer node IDs, 'weight' edge attribute, and
        optional node attributes (product_id, commodity_desc, etc.).
    """
    G = nx.DiGraph()

    lift_coo = lift_matrix.tocoo()
    for i, j, w in zip(lift_coo.row.tolist(), lift_coo.col.tolist(), lift_coo.data.tolist()):
        G.add_edge(int(i), int(j), weight=float(w))
        G.add_edge(int(j), int(i), weight=float(w))

    if product_meta is not None:
        for node in G.nodes():
            if node in product_meta.index:
                row = product_meta.loc[node]
                nx.set_node_attributes(G, {node: row.to_dict()})

    return G


def save_graph(G: nx.DiGraph, output_dir: str) -> None:
    """
    Serialize the graph to GraphML and GEXF formats.

    Both files are written to output_dir. GraphML is the canonical format
    (strict typing, weight preserved as float). GEXF is included for
    Gephi visualization and supports dynamic attributes if needed later.

    Args:
        G: directed graph with 'weight' edge attribute.
        output_dir: directory where files will be written.
    """
    os.makedirs(output_dir, exist_ok=True)
    nx.write_graphml(G, os.path.join(output_dir, "association_graph.graphml"))
    nx.write_gexf(G, os.path.join(output_dir, "association_graph.gexf"))


def load_graph(output_dir: str, fmt: str = "graphml") -> nx.DiGraph:
    """
    Load the graph from a serialized file and restore integer node IDs.

    NetworkX encodes node IDs as strings in GraphML/GEXF. This function
    relabels nodes back to integers after loading.

    Args:
        output_dir: directory containing the serialized graph files.
        fmt: 'graphml' (default) or 'gexf'.

    Returns:
        nx.DiGraph with integer node IDs and 'weight' edge attribute.
    """
    if fmt == "graphml":
        path = os.path.join(output_dir, "association_graph.graphml")
        G = nx.read_graphml(path)
    elif fmt == "gexf":
        path = os.path.join(output_dir, "association_graph.gexf")
        G = nx.read_gexf(path)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'graphml' or 'gexf'.")

    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    return G
