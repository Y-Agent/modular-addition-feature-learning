#!/bin/bash
# Pre-compute results for all odd p in [3, MAX_P].
# Deletes checkpoints after each p to save disk space.
#
# Usage:
#   bash precompute/run_all.sh          # p = 3, 5, 7, ..., 99
#   MAX_P=199 bash precompute/run_all.sh  # p = 3, 5, 7, ..., 199
#
# Run from the project root directory.

MAX_P=${MAX_P:-99}

set -e
echo "=== Pre-computing all odd p in [3, $MAX_P] ==="

COMPLETED=0
FAILED=0

for P in $(seq 3 2 "$MAX_P"); do
    echo ""
    echo "========================================"
    echo "  Processing p=$P"
    echo "========================================"
    if CLEANUP=1 bash precompute/run_pipeline.sh "$P"; then
        COMPLETED=$((COMPLETED + 1))
    else
        echo "[FAIL] p=$P failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== All done. Completed: $COMPLETED, Failed: $FAILED ==="
echo "=== Precomputed results size: ==="
du -sh precomputed_results/
