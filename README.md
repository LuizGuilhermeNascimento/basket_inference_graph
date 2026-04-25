# basket_inference_graph

Builds a weighted product-association graph from retail transaction data for basket inference tasks.

## Overview

Uses the [Dunnhumby — The Complete Journey](docs/dunnhumby%20-%20The%20Complete%20Journey%20User%20Guide.pdf) dataset. The pipeline:

1. **Preprocessing** — load, deduplicate, filter rare products, chronological train/test split
2. **Co-occurrence** — sparse matrix counting how often pairs of products appear in the same basket
3. **Lift** — normalizes co-occurrence by marginal frequencies: `lift(i,j) = C(i,j)·N / (N_i·N_j)`
4. **Backbone extraction** *(optional)* — prunes low-signal edges with the Disparity Filter (Serrano et al., 2009)
5. **Graph** — exports a weighted `DiGraph` in GraphML and GEXF formats

## Usage

```bash
python3 main.py \
  --data data/raw/transaction_data.parquet \
  --products data/raw/product.parquet \
  --output outputs/graphs
```

Key parameters:

| Flag | Default | Description |
|---|---|---|
| `--min-support` | 2 | Minimum basket appearances for a product to be kept |
| `--min-cooccurrence` | 2 | Minimum co-occurrence count for an edge |
| `--min-lift` | 0 | Minimum lift threshold |
| `--alpha` | 0 | Disparity filter threshold (0 = disabled) |
| `--train-fraction` | 0.8 | Fraction of days used for graph construction |

## Outputs

```
outputs/graphs/association_graph.graphml   # canonical format
outputs/graphs/association_graph.gexf      # Gephi-compatible
data/processed/transactions_processed.parquet
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis |
| `02_parameter_sensitivity.ipynb` | Effect of min-support, lift, and alpha on graph size |
| `03_backbone_analysis.ipynb` | Disparity filter behaviour and backbone structure |
| `04_graph_analysis.ipynb` | Degree distribution, centrality, community detection |

## Setup

```bash
pip install -r requirements.txt
```
