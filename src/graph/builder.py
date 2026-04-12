import os
import pandas as pd
import networkx as nx
import scipy.sparse as sp


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
