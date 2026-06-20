# Basket Inference Graph — MC859

Constrói um grafo de associação entre produtos a partir de dados de transações de varejo e avalia três métodos de previsão de cestas de compra incompletas.

## Problema

Dado que um cliente já colocou alguns itens no carrinho, quais outros itens ele provavelmente vai comprar? O pipeline transforma dados históricos de transações em um grafo de confiança entre produtos e usa esse grafo para recomendar itens faltantes.

## Dataset

[Dunnhumby — The Complete Journey](docs/dunnhumby%20-%20The%20Complete%20Journey%20User%20Guide.pdf) — histórico de compras de 2.500 domicílios em um período de dois anos.

Os arquivos brutos (`data/raw/`) devem estar no formato Parquet. Se você tiver os CSVs originais, converta-os primeiro:

```bash
python3 scripts/csv_to_parquet.py \
    --input-path  <diretório_com_csvs> \
    --output-path data/raw
```

## Configuração

```bash
pip install -r requirements.txt
```

## Passo 1 — Construir o grafo

```bash
python3 main.py \
    --data     data/raw/transaction_data.parquet \
    --products data/raw/product.parquet \
    --output   outputs/graphs
```

Este comando executa o pipeline completo de construção:

1. **Pré-processamento** — remove duplicatas e produtos com suporte abaixo do mínimo; faz divisão cronológica em treino/teste por dia.
2. **Co-ocorrência** — conta quantas cestas contêm cada par de produtos: `C[i,j]`.
3. **Confiança** — calcula arestas direcionadas: `conf(i→j) = C[i,j] / N_i`, onde `N_i` é o número de cestas com o produto `i`.
4. **Grafo** — exporta um `DiGraph` ponderado nos formatos GraphML e GEXF.

**Parâmetros principais:**

| Flag | Padrão | Descrição |
|---|---|---|
| `--data` | `data/raw/transaction_data.parquet` | Arquivo de transações |
| `--products` | `None` | Arquivo de produtos (opcional; adiciona atributos textuais aos nós) |
| `--output` | `outputs/graphs` | Diretório de saída do grafo |
| `--min-support` | `2` | Mínimo de cestas em que um produto deve aparecer |
| `--min-cooccurrence` | `2` | Mínimo de co-ocorrências para criar uma aresta |
| `--train-fraction` | `0.8` | Fração de dias usada para treino (os demais são teste) |

**Arquivos gerados:**

```
outputs/graphs/association_graph.graphml   # formato canônico
outputs/graphs/association_graph.gexf      # compatível com Gephi
outputs/graphs/item_counts.npy             # frequência por produto (usado no experimento)
data/processed/transactions_processed.parquet
data/processed/train.parquet
data/processed/test.parquet
```

## Passo 2 — Reproduzir o experimento

```bash
bash scripts/run_experiment.sh
```

O script executa o experimento principal (α = 0,85) e um varredura de α (0,30 / 0,50 / 0,70 / 0,85). O processamento é dividido em 5 *shards* sequenciais para reduzir uso de memória.

**Protocolo de avaliação — basket completion:**

Para cada cesta de teste:
1. Observa uma fração dos itens (25%, 50% ou 75%).
2. Tenta recomendar os itens restantes (ocultos).
3. Mede **Precision@k** e **Recall@k** para k ∈ {5, 10, 20}.
4. Repete 10 vezes com amostras diferentes do conjunto observado.

**Métodos comparados:**

| Método | Descrição |
|---|---|
| `popularity` | Ordena candidatos pela frequência global de treino — baseline sem contexto |
| `local_agg` | Média das arestas de confiança dos itens observados para cada candidato |
| `ppr` | Personalized PageRank sobre o grafo de confiança — propaga relevância por associações indiretas |

**Arquivos gerados:**

```
outputs/results/experiment_results_merged.parquet   # uma linha por (basket, n_obs, método, k, repetição)
outputs/results/product_hit_rates_merged.parquet    # taxa de acerto por produto
outputs/results/params_merged.json                  # parâmetros usados
outputs/results/alpha_sweep/alpha_<X>/              # resultados por valor de α
```

Para rodar apenas parte do experimento:

```bash
bash scripts/run_experiment.sh --skip-sweep   # apenas experimento principal
bash scripts/run_experiment.sh --skip-main    # apenas varredura de α
```

Para rodar um único shard manualmente:

```bash
python3 scripts/run_experiment.py \
    --n-obs 0.25 0.5 0.75 \
    --k-values 5 10 20 \
    --n-repetitions 10 \
    --min-basket-size 2 \
    --n-workers 4 \
    --n-shards 5 --shard-id 0 \
    --output outputs/results/experiment.parquet
```

Para mesclar shards após rodar manualmente:

```bash
python3 scripts/merge_shards.py \
    --results-dir outputs/results \
    --output      outputs/results/experiment_results_merged.parquet
```

## Passo 3 — Análise dos resultados

Abra `notebooks/experiment_analysis.ipynb` para visualizar:

- Recall@k e Precision@k por método e fração observada
- Curvas de desempenho em função de α (varredura PPR)
- Taxa de acerto por produto vs. centralidade no grafo
- Efeito do número de itens observados nas métricas

## Notebooks

| Notebook | Finalidade |
|---|---|
| `notebooks/eda.ipynb` | Análise exploratória das transações e produtos |
| `notebooks/graph_analysis.ipynb` | Estrutura do grafo: grau, centralidade, distribuição de pesos |
| `notebooks/experiment_analysis.ipynb` | Análise dos resultados do experimento de previsão |

## Estrutura do projeto

```
├── main.py                        # Passo 1: constrói o grafo
├── scripts/
│   ├── run_experiment.sh          # Passo 2: roda o experimento completo
│   ├── run_experiment.py          # Execução de um shard individual
│   ├── merge_shards.py            # Consolida resultados dos shards
│   └── csv_to_parquet.py          # Converte CSVs brutos para Parquet
├── src/
│   ├── preprocessing.py           # Carregamento, limpeza e split temporal
│   ├── graph_builder.py           # Co-ocorrência, confiança e I/O do grafo
│   ├── recommenders.py            # Popularity, LocalAgg, PPR
│   └── evaluation.py              # Protocolo de avaliação e métricas
├── notebooks/
│   ├── eda.ipynb
│   ├── graph_analysis.ipynb
│   └── experiment_analysis.ipynb
├── data/
│   ├── raw/                       # Arquivos Parquet originais (não modificar)
│   └── processed/                 # Gerado pelo main.py
├── outputs/
│   ├── graphs/                    # Grafo gerado pelo main.py
│   └── results/                   # Resultados gerados pelo experimento
└── requirements.txt
```
