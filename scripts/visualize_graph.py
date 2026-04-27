"""
Visualize a subsample of the association graph.

Sampling strategies:
  ego   — ego network around a random (or specified) seed node
  bfs   — BFS expansion from a seed up to a given radius
  random — random induced subgraph of N nodes

Usage examples:
    python3 scripts/visualize_graph.py --strategy ego --seed 1 --radius 2
    python3 scripts/visualize_graph.py --strategy bfs --seed 1 --radius 2
    python3 scripts/visualize_graph.py --strategy random --n-nodes 100
    python3 scripts/visualize_graph.py --strategy random --n-nodes 200 --output my_graph.png
"""

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import networkx as nx

from src.graph_builder import load_graph

GRAPH_DIR = "outputs/graphs"


def subsample_ego(G: nx.DiGraph, seed: int, radius: int) -> nx.DiGraph:
    neighbors = nx.ego_graph(G, seed, radius=radius, undirected=True)
    return G.subgraph(neighbors.nodes()).copy()


def subsample_bfs(G: nx.DiGraph, seed: int, radius: int) -> nx.DiGraph:
    nodes = nx.single_source_shortest_path_length(
        G.to_undirected(as_view=True), seed, cutoff=radius
    ).keys()
    return G.subgraph(nodes).copy()


def subsample_random(G: nx.DiGraph, n_nodes: int, seed: int | None) -> nx.DiGraph:
    rng = random.Random(seed)
    sampled = rng.sample(list(G.nodes()), min(n_nodes, G.number_of_nodes()))
    return G.subgraph(sampled).copy()


def node_label(G: nx.DiGraph, node: int) -> str:
    attrs = G.nodes[node]
    return attrs.get("sub_commodity_desc", str(node))


def draw(sub: nx.DiGraph, title: str, output: str | None) -> None:
    pos = nx.spring_layout(sub, seed=42, k=1.5)

    weights = [d["weight"] for _, _, d in sub.edges(data=True)]
    w_min, w_max = (min(weights), max(weights)) if weights else (0, 1)
    w_range = w_max - w_min or 1
    edge_alpha = [0.2 + 0.6 * (w - w_min) / w_range for w in weights]

    degrees = dict(sub.degree())
    node_size = [100 + 20 * degrees[n] for n in sub.nodes()]

    labels = {n: node_label(sub, n) for n in sub.nodes()}

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(sub, pos, node_size=node_size, node_color="steelblue", alpha=0.85, ax=ax)
    nx.draw_networkx_edges(
        sub, pos,
        alpha=edge_alpha,
        edge_color="gray",
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=7, ax=ax)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in sub.edges(data=True)}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    ax.set_title(
        f"{title}\n{sub.number_of_nodes()} nodes · {sub.number_of_edges()} edges",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default=GRAPH_DIR)
    parser.add_argument("--strategy", choices=["ego", "bfs", "random"], default="ego")
    parser.add_argument("--seed", type=int, default=None, help="Seed node (ego/bfs) or RNG seed (random)")
    parser.add_argument("--radius", type=int, default=2, help="Hop radius for ego/bfs")
    parser.add_argument("--n-nodes", type=int, default=150, help="Number of nodes for random strategy")
    parser.add_argument("--output", default=None, help="Save figure to this path instead of showing")
    args = parser.parse_args()

    print(f"Loading graph from {args.graph_dir} ...")
    G = load_graph(args.graph_dir, fmt="graphml")
    print(f"Graph: {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")

    if args.strategy in ("ego", "bfs"):
        seed_node = args.seed if args.seed is not None else random.choice(list(G.nodes()))
        if seed_node not in G:
            print(f"Node {seed_node} not in graph.", file=sys.stderr)
            sys.exit(1)
        print(f"Strategy: {args.strategy} | seed={seed_node} | radius={args.radius}")
        sub = (subsample_ego if args.strategy == "ego" else subsample_bfs)(G, seed_node, args.radius)
        title = f"{args.strategy.upper()} network — seed {seed_node} (r={args.radius})"
    else:
        print(f"Strategy: random | n_nodes={args.n_nodes} | rng_seed={args.seed}")
        sub = subsample_random(G, args.n_nodes, args.seed)
        title = f"Random subgraph ({args.n_nodes} nodes)"

    if sub.number_of_nodes() == 0:
        print("Empty subgraph — try a different seed or larger radius.", file=sys.stderr)
        sys.exit(1)

    print(f"Subgraph: {sub.number_of_nodes()} nodes · {sub.number_of_edges()} edges")
    draw(sub, title, args.output)


if __name__ == "__main__":
    main()
