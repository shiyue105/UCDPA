#!/usr/bin/env bash
# Auto-launcher: waits for CIFAR-100-C to finish on GPU 3, then launches
# ResNet-50 cross-backbone experiments on GPU 3.
#
# Usage:
#   nohup bash /data/yuehan/outputs/ucdpa_tta/auto_launch_resnet50.sh > \
#       /data/yuehan/outputs/ucdpa_tta/logs/_auto_launcher.log 2>&1 &

set -u  # Do NOT use -e: we want to keep polling even if intermediate checks fail.

PY=/data/yuehan/envs/tta_env/bin/python
PROJECT_ROOT=/data/yuehan/outputs/ucdpa_tta
WORKDIR="$PROJECT_ROOT/code"
CIFAR_LOG="$PROJECT_ROOT/logs/_cifar100_gpu3_v3.log"
RESNET_LOG="$PROJECT_ROOT/logs/_resnet50_gpu3.log"
ITEMS_FILE="$PROJECT_ROOT/_resnet50_gpu3.json"

# Sanity checks
if [ ! -f "$ITEMS_FILE" ]; then
    echo "[ERROR] ResNet-50 items file not found: $ITEMS_FILE"
    echo "        Run: cd $WORKDIR && $PY build_resnet50_items.py"
    exit 1
fi

# Number of items expected
EXPECTED=$(python3 -c "import json; print(len(json.load(open('$ITEMS_FILE'))))")
echo "[INFO] Auto-launcher started at $(date)"
echo "[INFO] Waiting for CIFAR-100-C (GPU 3) to complete before launching ResNet-50..."
echo "[INFO] Expected items on GPU 3: $EXPECTED"
echo "[INFO] Polling CIFAR log: $CIFAR_LOG"

# Polling loop: wait until the CIFAR-100-C process exits
while true; do
    # Check if the CIFAR-100-C worker is still running on GPU 3
    if ! pgrep -f "_cifar100_all_gpu3.json" > /dev/null 2>&1; then
        echo "[INFO] CIFAR-100-C worker has exited at $(date)"
        break
    fi
    # Show progress
    DONE_COUNT=$(grep -c "\[DONE\]" "$CIFAR_LOG" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] CIFAR-100-C progress: $DONE_COUNT / 133"
    sleep 60
done

# Final CIFAR count
FINAL_CIFAR=$(grep -c "\[DONE\]" "$CIFAR_LOG" 2>/dev/null || echo 0)
echo "[INFO] CIFAR-100-C final count: $FINAL_CIFAR / 133"

# Now launch ResNet-50 on GPU 3
echo "[INFO] Launching ResNet-50 worker on GPU 3 at $(date)"

cd "$WORKDIR"
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/data/yuehan/cache/huggingface
export TORCH_HOME=/data/yuehan/cache/torch
export XDG_CACHE_HOME=/data/yuehan/cache
export MPLCONFIGDIR=/data/yuehan/cache/matplotlib
export PYTHONPATH="$WORKDIR"

"$PY" run_experiment.py --worker \
    --items_file "$ITEMS_FILE" --gpu 3 \
    > "$RESNET_LOG" 2>&1

RESNET_EXIT=$?
echo "[INFO] ResNet-50 worker exited with code $RESNET_EXIT at $(date)"

# Final ResNet-50 count
RESNET_DONE=$(grep -c "\[DONE\]" "$RESNET_LOG" 2>/dev/null || echo 0)
echo "[INFO] ResNet-50 GPU 3 final count: $RESNET_DONE / $EXPECTED"
echo "[INFO] Auto-launcher finished at $(date)"
