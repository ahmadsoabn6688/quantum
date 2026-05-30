# Reproducibility Notes

## Hardware

The experiments were designed to run on CPU-only laptop hardware. No GPU is required.

## Software Environment

Conda environment name:

```bash
qchem-kernel

Main packages:

Python 3.10
RDKit
NumPy
Pandas
scikit-learn
XGBoost
PennyLane
Matplotlib
PyYAML
tqdm


## Dataset

The BACE dataset is downloaded from DeepChem:

wget -O data/raw/bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv

## Reproduce Full Experiment

Activate environment:

conda activate qchem-kernel

Run 	full pipeline:

./scripts/run_all.sh

## Main Output Files
results/tables/bace_combined_subset_results.csv
results/figures/bace_auroc_vs_runtime.png
results/figures/bace_ece_vs_runtime.png
manuscript/initial_results_bace.md
# Random Seed

All current experiments use:
seed = 0


## Current Experimental Setting
Dataset: BACE
Features: ECFP4 1024-bit + 8 physicochemical descriptors
PCA: 4 and 6 dimensions
Train subset: 300 molecules
Test subset: 100 molecules
Quantum models: fixed QSVM and anchor-based QSVM
Classical baselines: RBF-SVM, Random Forest, XGBoost
Metrics: Accuracy, F1, AUROC, AUPRC, ECE, runtime
