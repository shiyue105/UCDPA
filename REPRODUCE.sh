#!/usr/bin/env bash
# =============================================================================
# REPRODUCE.sh
# -----------------------------------------------------------------------------
# End-to-end reproduction script for the UCDPA (Uncertainty-Calibrated
# Dual-Path Adaptation) Test-Time Adaptation study on ImageNet-C ViT-B/16.
#
# Reproduces every reported result in order:
#   1. activate environment
#   2. set cache paths + GPU selection
#   3. environment check (python, torch, timm, dataset, model)
#   4. run smoke test
#   5. run full experiments   (full ImageNet-C streams, NO --max_batches)
#   6. aggregate metrics
#   7. generate figures
#
# Experiment grid: 7 methods x 19 corruptions x 5 severities x 3 seeds = 1995 runs.
# Expected wall-clock time: ~100 hours on 6 GPUs (CUDA devices 2,5,6,7,8,9).
#
# Usage:
#   bash /data/yuehan/outputs/ucdpa_tta/REPRODUCE.sh
# =============================================================================
set -euo pipefail

# Absolute path to the project venv interpreter (used for every step).
PY=/data/yuehan/envs/tta_env/bin/python
# Working directory that contains run_experiment.py, aggregate.py, etc.
WORKDIR=/data/yuehan/outputs/ucdpa_tta/code
PROJECT_ROOT=/data/yuehan/outputs/ucdpa_tta

# =============================================================================
# Step 1: activate environment
# =============================================================================
echo "============================================================"
echo "[Step 1/7] Activate environment"
echo "============================================================"
source /data/yuehan/envs/tta_env/bin/activate

# =============================================================================
# Step 2: set cache paths (and GPU selection)
# =============================================================================
echo "============================================================"
echo "[Step 2/7] Set cache paths and GPU selection"
echo "============================================================"
export HF_HOME=/data/yuehan/cache/huggingface
export TORCH_HOME=/data/yuehan/cache/torch
export TIMM_HOME=/data/yuehan/cache/timm
export XDG_CACHE_HOME=/data/yuehan/cache
export MPLCONFIGDIR=/data/yuehan/cache/matplotlib
# Pin the GPUs used by the full experiment (6 GPUs).
export CUDA_VISIBLE_DEVICES=2,5,6,7,8,9

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TIMM_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$WORKDIR"

# =============================================================================
# Step 3: environment check
# =============================================================================
echo "============================================================"
echo "[Step 3/7] Environment check"
echo "============================================================"
echo "Python: $($PY --version 2>&1)"
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())' 2>&1)"
echo "timm: $($PY -c 'import timm; print(timm.__version__)' 2>&1)"
echo "numpy: $($PY -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "pandas: $($PY -c 'import pandas; print(pandas.__version__)' 2>&1)"
echo "scikit-learn: $($PY -c 'import sklearn; print(sklearn.__version__)' 2>&1)"

# Verify dataset exists
DATA_ROOT=/data/share/datasets
if [ ! -d "$DATA_ROOT/imagenet-c" ]; then
    echo "ERROR: ImageNet-C dataset not found at $DATA_ROOT/imagenet-c"
    exit 1
fi
N_CORR=$(ls -d "$DATA_ROOT/imagenet-c"/*/ 2>/dev/null | wc -l)
echo "ImageNet-C: $N_CORR corruptions found at $DATA_ROOT/imagenet-c"

# Verify model exists
MODEL_DIR=/data/share/models/timm/vit_base_patch16_224
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: ViT-B/16 model directory not found at $MODEL_DIR"
    exit 1
fi
echo "ViT-B/16 model: found at $MODEL_DIR"
ls -la "$MODEL_DIR"/

# =============================================================================
# Step 4: run smoke test
# =============================================================================
echo "============================================================"
echo "[Step 4/7] Smoke test"
echo "============================================================"
# Tiny smoke test (2 batches internally) to validate code correctness before
# the long full run. Verifies: model loads, 1000-class head, all 7 methods
# complete one forward+backward, metrics aggregate, no --max_batches in full cmd.
"$PY" smoke_test.py

# =============================================================================
# Step 5: run full experiments (NO --max_batches)
# =============================================================================
echo "============================================================"
echo "[Step 5/12] Full experiments (full ImageNet-C streams, no --max_batches)"
echo "============================================================"
# 7 methods x 19 corruptions x 5 severities x 3 seeds = 1995 runs.
# NOTE: the final full experiment MUST NOT use --max_batches.
"$PY" run_experiment.py \
    --mode full \
    --gpus 2,5,6,7,8,9

# =============================================================================
# Step 5b: run supplementary experiments (ablations + sensitivity + batch size)
# =============================================================================
echo "============================================================"
echo "[Step 5b/12] Supplementary experiments (ablation + sensitivity + batch size)"
echo "============================================================"
# Builds and launches:
#   - ucdpa_noscalenorm: 95 runs (scale-norm ablation, seed 0)
#   - ucdpa_sens_{tauc,tauu,Tc,Tu}: 300 runs (sensitivity sweep, seed 0)
#   - ucdpa_bs{1,8,32,64}: 76 runs (batch size sensitivity, seed 0)
# Total: 471 runs across 6 GPUs.
"$PY" build_supplementary_items.py
for g in 2 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=$g \
    HF_HOME=$HF_HOME TORCH_HOME=$TORCH_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME \
    MPLCONFIGDIR=$MPLCONFIGDIR PYTHONPATH="$WORKDIR" \
    "$PY" run_experiment.py --worker \
        --items_file "$PROJECT_ROOT/_supp_gpu$g.json" --gpu $g \
        > "$PROJECT_ROOT/logs/_supp_gpu$g.log" 2>&1 &
done
wait

# =============================================================================
# Step 5c: run CIFAR-100-C cross-dataset experiments
# =============================================================================
echo "============================================================"
echo "[Step 5c/12] CIFAR-100-C experiments (ResNet-18, 133 runs)"
echo "============================================================"
# 7 methods x 19 corruptions x severity 3 x seed 0 = 133 runs on CIFAR-100-C
# Validates UCDPA on a different dataset + CNN backbone (BatchNorm adaptation).
"$PY" build_cifar100_items.py
for g in 2 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=$g \
    HF_HOME=$HF_HOME TORCH_HOME=$TORCH_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME \
    MPLCONFIGDIR=$MPLCONFIGDIR PYTHONPATH="$WORKDIR" \
    "$PY" run_experiment.py --worker \
        --items_file "$PROJECT_ROOT/_cifar100_gpu$g.json" --gpu $g \
        > "$PROJECT_ROOT/logs/_cifar100_gpu$g.log" 2>&1 &
done
wait

# =============================================================================
# Step 5d: run ResNet-50 cross-backbone experiments
# =============================================================================
echo "============================================================"
echo "[Step 5d/12] ResNet-50 experiments (ImageNet-C, 133 runs)"
echo "============================================================"
# 7 methods x 19 corruptions x severity 3 x seed 0 = 133 runs on ResNet-50
# Validates UCDPA on a CNN backbone with BatchNorm adaptation.
"$PY" build_resnet50_items.py
for g in 2 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=$g \
    HF_HOME=$HF_HOME TORCH_HOME=$TORCH_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME \
    MPLCONFIGDIR=$MPLCONFIGDIR PYTHONPATH="$WORKDIR" \
    "$PY" run_experiment.py --worker \
        --items_file "$PROJECT_ROOT/_resnet50_gpu$g.json" --gpu $g \
        > "$PROJECT_ROOT/logs/_resnet50_gpu$g.log" 2>&1 &
done
wait

# =============================================================================
# Step 6: aggregate metrics
# =============================================================================
echo "============================================================"
echo "[Step 6/12] Aggregate metrics"
echo "============================================================"
# Roll up per-run summaries into summary CSVs and comparison tables.
# Now generates 10 tables (6 main + 4 supplementary).
"$PY" aggregate.py

# =============================================================================
# Step 7: generate figures
# =============================================================================
echo "============================================================"
echo "[Step 7/9] Generate figures"
echo "============================================================"
"$PY" make_figures.py

# =============================================================================
# Step 8: bootstrap CI and statistical significance tests
# =============================================================================
echo "============================================================"
echo "[Step 8/9] Bootstrap CI and statistical significance tests"
echo "============================================================"
"$PY" bootstrap_ci.py

# =============================================================================
# Step 9: fill paper placeholders and compile
# =============================================================================
echo "============================================================"
echo "[Step 9/9] Generate paper (fill placeholders + compile)"
echo "============================================================"
"$PY" generate_paper.py

echo "============================================================"
echo "Reproduction complete. Expected outputs:"
echo "  results/summaries/full_summary.csv"
echo "  results/summaries/*.csv          (per-run summaries)"
echo "  results/samples/*.parquet        (per-sample diagnostics)"
echo "  results/tables/*.{csv,md,tex}    (6 comparison tables)"
echo "  results/figures/*.{pdf,png}      (7 figures)"
echo "  results/bootstrap_ci_summary.csv (statistical significance)"
echo "  paper/main.pdf                   (compiled paper with real values)"
echo "  logs/smoke_test.log"
echo "  logs/_worker_gpu*.log"
echo "============================================================"
