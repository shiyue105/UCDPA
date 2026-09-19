#!/usr/bin/env bash
# Final launcher: waits for supplementary experiments to finish, then
# launches ResNet-50 cross-backbone experiments on freed GPUs, and
# finally runs the aggregation pipeline.
#
# Usage:
#   nohup bash /data/yuehan/outputs/ucdpa_tta/final_launcher.sh > \
#       /data/yuehan/outputs/ucdpa_tta/logs/_final_launcher.log 2>&1 &

set -u

PY=/data/yuehan/envs/tta_env/bin/python
PROJECT_ROOT=/data/yuehan/outputs/ucdpa_tta
WORKDIR="$PROJECT_ROOT/code"
GPUS=("2" "5" "6" "7" "8" "9")

echo "[INFO] Final launcher started at $(date)"

# =============================================================================
# Phase 1: Wait for supplementary experiments to finish
# =============================================================================
echo "[INFO] Phase 1: Waiting for supplementary experiments to finish..."

while true; do
    RUNNING=0
    for g in "${GPUS[@]}"; do
        if pgrep -f "_supp_gpu$g.json" > /dev/null 2>&1; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    DONE_TOTAL=0
    for g in "${GPUS[@]}"; do
        c=$(grep -c "\[DONE\]" "$PROJECT_ROOT/logs/_supp_gpu$g.log" 2>/dev/null || echo 0)
        DONE_TOTAL=$((DONE_TOTAL + c))
    done
    echo "[$(date '+%H:%M:%S')] Supp workers running: $RUNNING/6, done: $DONE_TOTAL/471"
    if [ "$RUNNING" -eq 0 ]; then
        echo "[INFO] All supplementary workers finished at $(date)"
        echo "[INFO] Final supp count: $DONE_TOTAL / 471"
        break
    fi
    sleep 120
done

# =============================================================================
# Phase 2: Launch ResNet-50 on all 6 GPUs
# =============================================================================
echo "[INFO] Phase 2: Launching ResNet-50 on all 6 GPUs at $(date)"

cd "$WORKDIR"
export HF_HOME=/data/yuehan/cache/huggingface
export TORCH_HOME=/data/yuehan/cache/torch
export XDG_CACHE_HOME=/data/yuehan/cache
export MPLCONFIGDIR=/data/yuehan/cache/matplotlib
export PYTHONPATH="$WORKDIR"

# Also try GPU 3 if it's free (check memory)
GPU3_FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null | head -1)
if [ -n "$GPU3_FREE" ] && [ "$GPU3_FREE" -lt 8000 ]; then
    echo "[INFO] GPU 3 has only ${GPU3_FREE}MiB used — launching ResNet-50 on GPU 3 too"
    CUDA_VISIBLE_DEVICES=3 \
    "$PY" run_experiment.py --worker \
        --items_file "$PROJECT_ROOT/_resnet50_gpu3.json" --gpu 3 \
        > "$PROJECT_ROOT/logs/_resnet50_gpu3.log" 2>&1 &
    GPUS_WITH_3=("2" "3" "5" "6" "7" "8" "9")
else
    echo "[INFO] GPU 3 has ${GPU3_FREE}MiB used — skipping (will redistribute work)"
    GPUS_WITH_3=("2" "5" "6" "7" "8" "9")
fi

for g in 2 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=$g \
    HF_HOME=$HF_HOME TORCH_HOME=$TORCH_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME \
    MPLCONFIGDIR=$MPLCONFIGDIR PYTHONPATH="$WORKDIR" \
    "$PY" run_experiment.py --worker \
        --items_file "$PROJECT_ROOT/_resnet50_gpu$g.json" --gpu $g \
        > "$PROJECT_ROOT/logs/_resnet50_gpu$g.log" 2>&1 &
    echo "[INFO] Launched ResNet-50 on GPU $g (PID $!)"
done

echo "[INFO] Waiting for all ResNet-50 workers to finish..."
wait

# Final ResNet-50 count
TOTAL_RESNET=0
for g in "${GPUS_WITH_3[@]}"; do
    c=$(grep -c "\[DONE\]" "$PROJECT_ROOT/logs/_resnet50_gpu$g.log" 2>/dev/null || echo 0)
    echo "[INFO] GPU $g: $c done"
    TOTAL_RESNET=$((TOTAL_RESNET + c))
done
echo "[INFO] Total ResNet-50 done: $TOTAL_RESNET / 133"

# =============================================================================
# Phase 3: Run aggregation pipeline
# =============================================================================
echo "[INFO] Phase 3: Running aggregation pipeline at $(date)"

cd "$WORKDIR"

echo "[INFO] Running aggregate.py..."
"$PY" aggregate.py 2>&1 | tee "$PROJECT_ROOT/logs/_final_aggregate.log"

echo "[INFO] Running make_figures.py..."
"$PY" make_figures.py 2>&1 | tee "$PROJECT_ROOT/logs/_final_figures.log"

echo "[INFO] Running generate_paper.py..."
"$PY" generate_paper.py 2>&1 | tee "$PROJECT_ROOT/logs/_final_paper.log"

echo "[INFO] Final launcher complete at $(date)"
echo "[INFO] All results in: $PROJECT_ROOT/results/"
echo "[INFO] Tables in:      $PROJECT_ROOT/results/tables/"
echo "[INFO] Figures in:     $PROJECT_ROOT/results/figures/"
echo "[INFO] Paper in:       $PROJECT_ROOT/paper/main.pdf"
