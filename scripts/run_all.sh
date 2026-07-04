#!/usr/bin/env bash
set -e

echo "=== Preparing BACE molecular features ==="
PYTHONPATH=. python scripts/prepare_bace_features.py

echo "=== Preparing PCA compressed features ==="
PYTHONPATH=. python scripts/prepare_bace_pca.py

echo "=== Running classical subset baselines ==="
PYTHONPATH=. python scripts/run_bace_subset_baselines.py

echo "=== Running full 4-qubit QSVM ==="
PYTHONPATH=. python scripts/run_bace_quantum_kernel.py

echo "=== Running 4-qubit quantum anchor model, 64 anchors ==="
PYTHONPATH=. python scripts/run_bace_quantum_anchor.py

echo "=== Running 4-qubit quantum anchor model, 128 anchors ==="
PYTHONPATH=. python scripts/run_bace_quantum_anchor_128.py

echo "=== Running 6-qubit quantum anchor model, 64 anchors ==="
PYTHONPATH=. python scripts/run_bace_quantum_anchor_6q_64.py

echo "=== Combining results ==="
PYTHONPATH=. python scripts/combine_bace_results.py

echo "=== Creating figures ==="
PYTHONPATH=. python scripts/plot_bace_results.py
PYTHONPATH=. python scripts/plot_bace_ece_runtime.py

echo "=== Printing summary ==="
PYTHONPATH=. python scripts/print_summary.py

echo "=== Done ==="
