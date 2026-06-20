import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def sample_observed(
    basket: set[int],
    n_obs: int,
    rng: np.random.Generator,
) -> tuple[set[int], set[int]]:
    """Returns (observed, hidden). Raises if |basket| <= n_obs."""
    if len(basket) <= n_obs:
        raise ValueError(f"Cannot sample {n_obs} observed from basket of size {len(basket)}")
    items = list(basket)
    observed = set(int(x) for x in rng.choice(items, size=n_obs, replace=False))
    hidden = basket - observed
    return observed, hidden


def compute_metrics(
    recommendations: list[int],
    hidden: set[int],
    k_values: list[int],
) -> dict[str, float]:
    """Returns {precision@k: float, recall@k: float} for each k in k_values."""
    metrics: dict[str, float] = {}
    for k in k_values:
        hits = len(set(recommendations[:k]) & hidden)
        precision = hits / k
        recall = hits / len(hidden) if hidden else 0.0
        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
    return metrics


def _build_dfs(
    records: list[dict],
    hit_stats: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = pd.DataFrame(records)
    hit_records = [
        {
            "product_idx": product,
            "method": method,
            "k": k,
            "n_hidden": counts[0],
            "n_hit": counts[1],
        }
        for product, methods in hit_stats.items()
        for method, ks in methods.items()
        for k, counts in ks.items()
    ]
    hit_stats_df = pd.DataFrame(hit_records)
    return results_df, hit_stats_df


def _resolve_n_obs(
    n_obs_specs: list[int | float | str], basket_size: int
) -> list[tuple[int | float, int]]:
    """Expand n_obs specs into (label, concrete_n_obs) pairs for a given basket size.

    Supported specs:
      - float 0 < p < 1: n_obs = ceil(p * basket_size); label = p (e.g. 0.25, 0.5, 0.75)
      - int or numeric string >= 1: fixed n_obs; label = int value
      - "half": alias for 0.5 (requires basket_size >= 6)
      - "all": every value 1 to basket_size - 1; labels are ints
    Pairs where concrete_n_obs >= basket_size or with duplicate concrete values are dropped.
    """
    seen: set[int] = set()
    result: list[tuple[int | float, int]] = []

    for spec in n_obs_specs:
        if spec == "half":
            pairs: list[tuple[int | float, int]] = (
                [(0.5, math.ceil(0.5 * basket_size))] if basket_size >= 6 else []
            )
        elif spec == "all":
            pairs = [(n, n) for n in range(1, basket_size)]
        else:
            try:
                val = float(spec)
            except (ValueError, TypeError):
                continue
            if 0 < val < 1:
                pairs = [(val, math.ceil(val * basket_size))]
            else:
                n = int(val)
                pairs = [(n, n)]

        for label, n in pairs:
            if n < basket_size and n not in seen:
                result.append((label, n))
                seen.add(n)

    return result


def min_in_graph_size_for_specs(n_obs_specs: list[int | float | str]) -> int:
    """Smallest |basket_in_graph| for which every spec in n_obs_specs produces a valid,
    distinct n_obs value — guaranteeing all n_obs conditions evaluate the same basket population.

    Returns 2 when "all" is present (the spec varies by design, population issue is inherent).
    """
    if "all" in n_obs_specs:
        return 2
    # At large n every fraction/int produces a distinct valid value — that's the target count.
    target = len(_resolve_n_obs(n_obs_specs, 10_000))
    for n in range(2, 10_001):
        if len(_resolve_n_obs(n_obs_specs, n)) == target:
            return n
    raise ValueError(f"Could not resolve min_in_graph_size for n_obs_specs={n_obs_specs!r}")


def _process_basket(
    basket_id: int,
    basket: set[int],
    basket_seed: int,
    graph_products: set[int],
    graph_products_arr: np.ndarray,
    recommenders: dict[str, Any],
    n_obs_values: list[int | str],
    k_values: list[int],
    n_repetitions: int,
    max_k: int,
    n_hidden: int | None = None,
    min_in_graph_size: int = 2,
) -> tuple[list[dict], dict]:
    basket_in_graph = basket & graph_products

    if n_hidden is not None:
        # Fixed-hidden protocol: hide exactly n_hidden items per rep, vary observed context.
        # recall denominator = n_hidden (constant), enabling detection of context utilization.
        if len(basket_in_graph) <= n_hidden:
            return [], {}

        rng = np.random.default_rng(basket_seed)
        basket_list = list(basket_in_graph)
        rep_seeds = rng.integers(0, 2**31, size=n_repetitions)
        records: list[dict] = []
        hit_stats: dict = {}

        for rep in range(n_repetitions):
            rep_rng = np.random.default_rng(int(rep_seeds[rep]))
            hidden = set(int(x) for x in rep_rng.choice(basket_list, size=n_hidden, replace=False))
            remaining = basket_in_graph - hidden
            remaining_list = list(remaining)

            for n_obs_label, n_obs in _resolve_n_obs(n_obs_values, len(remaining)):
                observed = set(int(x) for x in rep_rng.choice(remaining_list, size=n_obs, replace=False))
                candidates = graph_products - observed

                for method_name, rec in recommenders.items():
                    recs = rec.recommend(observed, candidates, max_k)
                    for k in k_values:
                        top_k_set = set(recs[:k])
                        n_hits = len(top_k_set & hidden)

                        for product in hidden:
                            if product not in hit_stats:
                                hit_stats[product] = {}
                            pm = hit_stats[product]
                            if method_name not in pm:
                                pm[method_name] = {}
                            pk = pm[method_name]
                            if k not in pk:
                                pk[k] = [0, 0]
                            pk[k][0] += 1
                            if product in top_k_set:
                                pk[k][1] += 1

                        records.append({
                            "basket_id": basket_id,
                            "n_obs": n_obs_label,
                            "method": method_name,
                            "k": k,
                            "repetition": rep,
                            "precision": n_hits / k,
                            "recall": n_hits / n_hidden,
                        })

        return records, hit_stats

    # --- Fraction protocol (original behaviour) ---
    if len(basket_in_graph) < min_in_graph_size:
        return [], {}

    rng = np.random.default_rng(basket_seed)
    records: list[dict] = []
    # hit_stats[product][method][k] = [n_hidden, n_hit]
    hit_stats: dict = {}

    ppr = recommenders.get("ppr")
    has_batch_ppr = ppr is not None and hasattr(ppr, "recommend_batch")

    for n_obs_label, n_obs in _resolve_n_obs(n_obs_values, len(basket_in_graph)):
        rep_seeds = rng.integers(0, 2**31, size=n_repetitions)

        # Pre-sample all reps so PPR can batch them
        observed_list: list[set[int]] = []
        hidden_list: list[set[int]] = []
        for rep in range(n_repetitions):
            rep_rng = np.random.default_rng(int(rep_seeds[rep]))
            observed, hidden = sample_observed(basket_in_graph, n_obs, rep_rng)
            observed_list.append(observed)
            hidden_list.append(hidden)

        # Batch PPR across all reps (one matmul per iteration instead of n_reps matvecs)
        if has_batch_ppr:
            ppr_recs_batch = ppr.recommend_batch(observed_list, graph_products_arr, max_k)
        else:
            ppr_recs_batch = None

        for rep in range(n_repetitions):
            observed = observed_list[rep]
            hidden = hidden_list[rep]
            candidates = graph_products - observed

            rep_recs: dict[str, list[int]] = {}
            for method_name, rec in recommenders.items():
                if method_name == "ppr" and ppr_recs_batch is not None:
                    rep_recs[method_name] = ppr_recs_batch[rep]
                else:
                    rep_recs[method_name] = rec.recommend(observed, candidates, max_k)

            for method_name, recs in rep_recs.items():
                for k in k_values:
                    top_k_set = set(recs[:k])
                    n_hits = len(top_k_set & hidden)

                    for product in hidden:
                        if product not in hit_stats:
                            hit_stats[product] = {}
                        pm = hit_stats[product]
                        if method_name not in pm:
                            pm[method_name] = {}
                        pk = pm[method_name]
                        if k not in pk:
                            pk[k] = [0, 0]
                        pk[k][0] += 1
                        if product in top_k_set:
                            pk[k][1] += 1

                    records.append({
                        "basket_id": basket_id,
                        "n_obs": n_obs_label,
                        "method": method_name,
                        "k": k,
                        "repetition": rep,
                        "precision": n_hits / k,
                        "recall": n_hits / len(hidden),
                    })

    return records, hit_stats


def run_basket_completion_experiment(
    test_baskets: list[set[int]],
    graph_products: set[int],
    recommenders: dict[str, Any],
    n_obs_values: list[int | str],
    k_values: list[int],
    n_repetitions: int,
    min_basket_size: int,
    max_basket_size: int | None = None,
    seed: int = 42,
    checkpoint_every: int = 0,
    checkpoint_fn: Callable[[pd.DataFrame, pd.DataFrame], None] | None = None,
    n_workers: int = 1,
    n_hidden: int | None = None,
    min_in_graph_basket_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the full basket completion experiment.

    Args:
        n_workers: number of threads for parallel basket processing.
                   scipy sparse matmuls release the GIL, so threading scales well.
        checkpoint_every: save partial results every N baskets evaluated (0 = disabled).
        checkpoint_fn: called with (partial_results_df, partial_hit_stats_df) at each checkpoint.
        min_in_graph_basket_size: minimum |basket_in_graph| for a basket to be evaluated.
            None (default) auto-derives from n_obs_values so that every n_obs spec produces a
            valid, distinct observed-set size — ensuring all conditions evaluate the same basket
            population. Ignored when n_hidden is not None (fixed-hidden protocol has its own check).

    Returns:
        results_df: one row per (basket_id, n_obs, method, k, repetition)
        hit_stats_df: per-product hit counts aggregated over all baskets/reps
    """
    master_rng = np.random.default_rng(seed)
    basket_seeds = master_rng.integers(0, 2**31, size=len(test_baskets))

    # Pre-convert graph_products to array once for batch PPR
    graph_products_arr = np.array(sorted(graph_products), dtype=np.intp)

    max_k = max(k_values)

    # For the fraction protocol, ensure all n_obs conditions evaluate the same basket population.
    # Fixed-hidden protocol uses its own size check (len(basket_in_graph) <= n_hidden).
    if n_hidden is None:
        effective_min_in_graph = (
            min_in_graph_basket_size
            if min_in_graph_basket_size is not None
            else min_in_graph_size_for_specs(n_obs_values)
        )
    else:
        effective_min_in_graph = 2

    records: list[dict] = []
    # hit_stats[product][method][k] = [n_hidden, n_hit]
    hit_stats: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))

    eligible = [
        (bid, basket)
        for bid, basket in enumerate(test_baskets)
        if len(basket) >= min_basket_size
        and (max_basket_size is None or len(basket) <= max_basket_size)
    ]

    def submit_one(bid: int, basket: set[int]):
        return _process_basket(
            bid, basket, int(basket_seeds[bid]),
            graph_products, graph_products_arr,
            recommenders, n_obs_values, k_values,
            n_repetitions, max_k, n_hidden,
            effective_min_in_graph,
        )

    n_evaluated = 0

    if n_workers == 1:
        it = iter(eligible)
        if _HAS_TQDM:
            it = _tqdm(it, total=len(eligible), desc="Baskets", unit="basket")
        for bid, basket in it:
            basket_records, basket_hit = submit_one(bid, basket)
            records.extend(basket_records)
            for product, methods in basket_hit.items():
                for method, ks in methods.items():
                    for k, counts in ks.items():
                        hit_stats[product][method][k][0] += counts[0]
                        hit_stats[product][method][k][1] += counts[1]
            n_evaluated += 1
            if checkpoint_every > 0 and checkpoint_fn and n_evaluated % checkpoint_every == 0:
                checkpoint_fn(*_build_dfs(records, hit_stats))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(submit_one, bid, basket): bid
                for bid, basket in eligible
            }
            completed_iter = as_completed(futures)
            if _HAS_TQDM:
                completed_iter = _tqdm(completed_iter, total=len(futures), desc="Baskets", unit="basket")
            for future in completed_iter:
                basket_records, basket_hit = future.result()
                records.extend(basket_records)
                for product, methods in basket_hit.items():
                    for method, ks in methods.items():
                        for k, counts in ks.items():
                            hit_stats[product][method][k][0] += counts[0]
                            hit_stats[product][method][k][1] += counts[1]
                n_evaluated += 1
                if checkpoint_every > 0 and checkpoint_fn and n_evaluated % checkpoint_every == 0:
                    checkpoint_fn(*_build_dfs(records, hit_stats))

    return _build_dfs(records, hit_stats)
