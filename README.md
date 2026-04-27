# Projeto de MC859 - Previsão de Cestas de Compra a partir de Grafos de Associação entre Produtos

Constrói um grafo ponderado de associação entre produtos a partir de dados de transações de varejo para tarefas de inferência de cesta.

## Visão geral

Utiliza o conjunto de dados [Dunnhumby — The Complete Journey](docs/dunnhumby%20-%20The%20Complete%20Journey%20User%20Guide.pdf). O pipeline é composto por:

1. **Pré-processamento** — carrega, filtra produtos raros e faz divisão cronológica em treino/teste
2. **Coocorrência** — matriz esparsa que conta com que frequência pares de produtos aparecem na mesma cesta
3. **Confiança** — calcula arestas direcionadas com `conf(i→j) = C(i,j) / N_i`
4. **Filtro por lift** — aplica limiar de lift `lift(i,j) = C(i,j)·N / (N_i·N_j)` durante a construção das arestas
5. **Grafo** — exporta um `DiGraph` ponderado (peso = confiança) nos formatos GraphML e GEXF

As instâncias do grafo estão compactadas em [instances.zip](instances.zip).

## Uso

```bash
python3 main.py \
  --data data/raw/transaction_data.parquet \
  --output outputs/graphs \
  --products data/raw/product.parquet
```

Parâmetros principais:

| Flag | Padrão | Descrição |
|---|---|---|
| `--data` | `data/raw/transaction_data.parquet` | Caminho do arquivo de transações |
| `--products` | `None` | Caminho do arquivo de produtos (opcional; habilita atributos de nós) |
| `--output` | `outputs/graphs` | Diretório para os arquivos do grafo |
| `--processed-output` | `data/processed/transactions_processed.parquet` | Arquivo parquet das transações processadas |
| `--train-output` | `data/processed/train.parquet` | Arquivo parquet da partição de treino |
| `--test-output` | `data/processed/test.parquet` | Arquivo parquet da partição de teste |
| `--min-support` | 2 | Número mínimo de aparições em cestas para manter um produto |
| `--min-confidence` | 0.0 | Limiar mínimo de confiança para manter arestas direcionadas |
| `--min-cooccurrence` | 2 | Contagem mínima de coocorrência para manter uma aresta |
| `--min-lift` | 1 | Limiar mínimo de lift usado como filtro |
| `--train-fraction` | 0.8 | Fração de dias usada para a construção do grafo |

## Saídas

```
outputs/graphs/association_graph.graphml   # formato canônico
outputs/graphs/association_graph.gexf      # compatível com Gephi
data/processed/transactions_processed.parquet
data/processed/train.parquet
data/processed/test.parquet
```

## Notebooks

| Notebook | Finalidade |
|---|---|
| `notebooks/eda.ipynb` | Análise exploratória dos dados de transações e produtos |
| `notebooks/graph_analysis.ipynb` | Análise estrutural do grafo (grau, centralidade e comunidades) |

## Visualização

Para visualizar subgrafos do resultado:

```bash
python3 scripts/visualize_graph.py --strategy bfs --seed 1 --radius 1 --n-nodes 50 --output my_graph.png
```

Parâmetros de visualização:

| Flag | Padrão | Descrição |
|---|---|---|
| `--graph-dir` | `outputs/graphs` | Diretório contendo `association_graph.graphml` |
| `--strategy` | `ego` | Estratégia de amostragem: `ego`, `bfs` ou `random` |
| `--seed` | `None` | Nó semente (`ego`/`bfs`) ou semente do RNG (`random`) |
| `--radius` | 2 | Raio (saltos) para `ego`/`bfs` |
| `--n-nodes` | 150 | Número de nós para a estratégia `random` |
| `--output` | `None` | Caminho de saída da figura; se omitido, abre visualização interativa |

## Configuração

```bash
pip install -r requirements.txt
```
