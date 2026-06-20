#!/usr/bin/env bash
# Basket completion experiment — main run + PPR alpha sweep.
#
# Usage:
#   bash scripts/run_experiment.sh                # main experiment + alpha sweep
#   bash scripts/run_experiment.sh --skip-main    # alpha sweep only
#   bash scripts/run_experiment.sh --skip-sweep   # main experiment only

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
N_SHARDS=5
N_WORKERS=4
N_REPS=10
MIN_BASKET_SIZE=2
CHECKPOINT_EVERY=200
N_OBS="0.25 0.5 0.75"
K_VALUES="5 10 20"

DEFAULT_ALPHA=0.85
SWEEP_ALPHAS=(0.30 0.50 0.70 0.85)

SKIP_MAIN=false
SKIP_SWEEP=false
for arg in "$@"; do
    case "$arg" in
        --skip-main)  SKIP_MAIN=true  ;;
        --skip-sweep) SKIP_SWEEP=true ;;
    esac
done

# ── Helper: run all shards sequentially then merge ────────────────────────────
# Args: <results_dir> <output_parquet> <alpha>
#
# run_experiment.py places each shard at:
#   <dirname(output_parquet)>/shard_NN/<basename(output_parquet)>
# merge_shards.py collects shard_* dirs inside <results_dir>.
run_and_merge() {
    local results_dir="$1"
    local output_parquet="$2"
    local alpha="$3"

    for shard in $(seq 0 $((N_SHARDS - 1))); do
        echo "  shard $((shard + 1))/$N_SHARDS ..."
        python3 scripts/run_experiment.py \
            --n-obs $N_OBS \
            --k-values $K_VALUES \
            --min-basket-size "$MIN_BASKET_SIZE" \
            --n-repetitions  "$N_REPS" \
            --n-workers      "$N_WORKERS" \
            --checkpoint-every "$CHECKPOINT_EVERY" \
            --n-shards  "$N_SHARDS" \
            --shard-id  "$shard" \
            --alpha     "$alpha" \
            --output    "$output_parquet"
    done

    echo "  merging ..."
    python3 scripts/merge_shards.py \
        --results-dir "$results_dir" \
        --output      "$results_dir/experiment_results_merged.parquet"
    echo "  → $results_dir/experiment_results_merged.parquet"
}

# ── Main experiment (alpha = 0.85, same layout as before) ─────────────────────
if ! $SKIP_MAIN; then
    echo "=== Main experiment (alpha=$DEFAULT_ALPHA) ==="
    run_and_merge \
        "outputs/results" \
        "outputs/results/experiment.parquet" \
        "$DEFAULT_ALPHA"
fi

# ── Alpha sweep (PPR only varies; LocalAgg and Popularity results are reusable) ─
if ! $SKIP_SWEEP; then
    echo ""
    echo "=== Alpha sweep: ${SWEEP_ALPHAS[*]} ==="

    for alpha in "${SWEEP_ALPHAS[@]}"; do
        alpha_tag="${alpha/./_}"          # 0.30 → 0_30
        alpha_dir="outputs/results/alpha_sweep/alpha_${alpha_tag}"

        echo ""
        echo "--- alpha=$alpha → $alpha_dir ---"
        run_and_merge \
            "$alpha_dir" \
            "$alpha_dir/experiment.parquet" \
            "$alpha"
    done
fi

echo ""
echo "=== All done. ==="
