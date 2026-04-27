import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from dataclasses import dataclass


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


def compute_confidence(
    cooc: CooccurrenceData,
    min_cooccurrence: int = 5,
) -> sp.coo_matrix:
    """
    Compute confidence for all directed item pairs with sufficient co-occurrence.

    conf(i→j) = C[i,j] / N_items[i]

    Each undirected pair (i,j) produces two directed edges with (typically)
    different confidence weights, making the resulting matrix asymmetric.

    Args:
        cooc: output of build_cooccurrence.
        min_cooccurrence: discard pairs with fewer than this many co-occurrences.

    Returns:
        Sparse asymmetric COO matrix of confidence values (shape n×n).
    """
    C_upper = sp.triu(cooc.C, k=1).tocoo()

    mask = C_upper.data >= min_cooccurrence
    rows = C_upper.row[mask]
    cols = C_upper.col[mask]
    c_vals = C_upper.data[mask].astype(np.float64)

    conf_ij = c_vals / cooc.N_items[rows]
    conf_ji = c_vals / cooc.N_items[cols]

    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_weights = np.concatenate([conf_ij, conf_ji])

    return sp.coo_matrix(
        (all_weights, (all_rows.astype(np.int32), all_cols.astype(np.int32))),
        shape=(cooc.n_products, cooc.n_products),
    )


# ---------------------------------------------------------------------------
# Graph assembly and I/O
# ---------------------------------------------------------------------------

def build_graph(
    weight_matrix: sp.coo_matrix,
    product_meta: pd.DataFrame | None = None,
) -> nx.DiGraph:
    """
    Assemble a weighted directed graph from a COO weight matrix.

    Each nonzero entry (i, j) produces one directed edge i→j with the
    given weight. For asymmetric metrics like confidence, (i,j) and (j,i)
    carry different weights and must both be present in weight_matrix.

    If product_meta is provided (DataFrame indexed by product_idx with columns
    like commodity_desc, department, brand), those fields are attached as node
    attributes — useful for Gephi visualisation and analysis notebooks.

    Args:
        weight_matrix: sparse COO matrix with edge weights.
        product_meta: optional metadata DataFrame from clean_transactions.

    Returns:
        nx.DiGraph with integer node IDs, 'weight' edge attribute, and
        optional node attributes (product_id, commodity_desc, etc.).
    """
    G = nx.DiGraph()

    weight_coo = weight_matrix.tocoo()
    for i, j, w in zip(weight_coo.row.tolist(), weight_coo.col.tolist(), weight_coo.data.tolist()):
        G.add_edge(int(i), int(j), weight=float(w))

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
