#!/bin/bash
# Full pre-computation pipeline for a single modulus p (any odd number >= 3).
#
# Usage:
#   bash precompute/run_pipeline.sh 23
#   bash precompute/run_pipeline.sh 9 --d_mlp 128
#   P=23 bash precompute/run_pipeline.sh
#
#   # Delete checkpoints after generating plots (saves disk space):
#   CLEANUP=1 bash precompute/run_pipeline.sh 97
#
# Run from the project root directory.

P=${1:-${P:-23}}
shift 2>/dev/null || true  # consume the p arg

# CLEANUP=1 to delete model checkpoints after plot generation
CLEANUP=${CLEANUP:-0}

# Collect remaining args (e.g. --d_mlp 128) to pass to train_all.py
EXTRA_ARGS="$@"

set -e
echo "=== Running full pipeline for p=$P $EXTRA_ARGS ==="

# Step 1: Train all 5 configurations
echo ""
echo "--- Step 1/4: Training ---"
python precompute/train_all.py --p "$P" --output ./trained_models --resume $EXTRA_ARGS

# Step 2: Generate model-based plots (d_mlp inferred from checkpoint)
echo ""
echo "--- Step 2/4: Generating model-based plots ---"
python precompute/generate_plots.py --p "$P" --input ./trained_models --output ./precomputed_results

# Step 3: Generate analytical simulation plots
echo ""
echo "--- Step 3/4: Generating analytical plots ---"
python precompute/generate_analytical.py --p "$P" --output ./precomputed_results

# Step 4: Cleanup checkpoints if requested
PADDED=$(printf '%03d' "$P")
MODEL_DIR="trained_models/p_${PADDED}"
if [ "$CLEANUP" = "1" ] && [ -d "$MODEL_DIR" ]; then
    echo ""
    echo "--- Cleanup: Deleting checkpoints for p=$P ---"
    SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
    rm -rf "$MODEL_DIR"
    echo "    Freed $SIZE from $MODEL_DIR"
fi

# Step 5: Verify
echo ""
echo "--- Verification ---"
RESULT_DIR="precomputed_results/p_${PADDED}"
echo "=== Results in ${RESULT_DIR}/ ==="
ls -la "${RESULT_DIR}/"
FILE_COUNT=$(ls -1 "${RESULT_DIR}/" | wc -l | tr -d ' ')
echo "=== Total files: ${FILE_COUNT} ==="
echo "=== Pipeline complete for p=$P ==="
