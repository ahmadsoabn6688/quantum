# Quantum Chemistry Kernel

This repository implements a lightweight hybrid quantum-classical pipeline for molecular classification using chemistry-aware molecular features and small quantum kernel models.

The initial experiment focuses on the BACE molecular classification dataset and investigates whether small quantum kernel models can be practically integrated into a reproducible cheminformatics workflow using CPU-only laptop hardware.

The goal of this project is **not** to claim quantum advantage. Instead, the aim is to study the feasibility, reproducibility, runtime cost, and predictive behavior of small quantum kernels when combined with classical molecular representations.

## Current Pipeline

```text
SMILES molecules
→ ECFP4 fingerprint + physicochemical descriptors
→ Standardization
→ PCA compression
→ Classical baselines and quantum kernel models
→ Performance, calibration, and runtime evaluation
```

## Dataset

The current experiment uses the **BACE** molecular classification dataset.

* Task: binary molecular classification
* Number of valid molecules: 1513
* Input: SMILES strings
* Output: binary activity label

The dataset is downloaded from DeepChem:

```bash
wget -O data/raw/bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv
```

## Molecular Features

Each molecule is represented using a chemistry-aware feature vector composed of:

* ECFP4 fingerprint, 1024 bits
* Molecular weight
* LogP
* Number of hydrogen bond donors
* Number of hydrogen bond acceptors
* Topological polar surface area
* Number of rotatable bonds
* Heavy atom count
* Number of rings

Total feature dimension:

```text
1024 + 8 = 1032 features
```

## Dimensionality Reduction

The 1032-dimensional molecular representation is standardized and compressed using PCA.

Current PCA settings:

```text
PCA-4: used for 4-qubit quantum models
PCA-6: used for 6-qubit quantum models
```

This allows the compressed molecular representation to be directly encoded into small quantum feature maps.

## Models

### Classical Baselines

The following classical models are evaluated using the same PCA-compressed molecular features:

* RBF-SVM
* Random Forest
* XGBoost

### Quantum Models

The following quantum kernel models are evaluated:

* Fixed 4-qubit QSVM
* 4-qubit anchor-based quantum SVM with 64 anchors
* 4-qubit anchor-based quantum SVM with 128 anchors
* 6-qubit anchor-based quantum SVM with 64 anchors

The anchor-based models approximate the full quantum kernel by representing each molecule through its quantum kernel similarities to a smaller set of anchor molecules.

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* F1 score
* AUROC
* AUPRC
* Expected Calibration Error
* Runtime in seconds

Expected Calibration Error is included to measure whether model confidence aligns with predictive correctness.

## Initial Experimental Setting

The current pilot experiment uses:

```text
Dataset: BACE
Train subset: 300 molecules
Test subset: 100 molecules
Seed: 0
Hardware: CPU-only laptop
```

## Current Results Summary

| Model                | PCA | Anchors | Accuracy |    F1 | AUROC | AUPRC |   ECE | Runtime (s) |
| -------------------- | --: | ------: | -------: | ----: | ----: | ----: | ----: | ----------: |
| RBF-SVM              |   4 |       - |    0.690 | 0.667 | 0.708 | 0.682 | 0.080 |       0.009 |
| Random Forest        |   4 |       - |    0.720 | 0.696 | 0.786 | 0.770 | 0.071 |       0.254 |
| XGBoost              |   4 |       - |    0.710 | 0.674 | 0.751 | 0.768 | 0.122 |       0.414 |
| QSVM-fixed           |   4 |    full |    0.700 | 0.667 | 0.714 | 0.699 | 0.092 |     280.254 |
| QAnchor-SVM-fixed    |   4 |      64 |    0.660 | 0.614 | 0.690 | 0.678 | 0.071 |      93.214 |
| QAnchor-SVM-fixed    |   4 |     128 |    0.700 | 0.659 | 0.708 | 0.700 | 0.074 |     185.883 |
| QAnchor-SVM-fixed-6q |   6 |      64 |    0.680 | 0.628 | 0.752 | 0.743 | 0.106 |     144.683 |

## Key Findings

The full 4-qubit QSVM achieved performance comparable to the RBF-SVM baseline on the same PCA-compressed molecular representation.

The 128-anchor 4-qubit quantum approximation preserved most of the full QSVM performance while reducing runtime.

The 6-qubit anchor-based quantum model achieved the best quantum AUROC and AUPRC, suggesting that increasing the quantum feature dimension can improve ranking-based molecular classification performance, although with higher runtime and calibration error.

The Random Forest baseline remained the strongest overall classical model in this pilot experiment.

## Reproducibility

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate qchem-kernel
```

To reproduce the full pipeline:

```bash
./scripts/run_all.sh
```

Main output files:

```text
results/tables/bace_combined_subset_results.csv
results/figures/bace_auroc_vs_runtime.png
results/figures/bace_ece_vs_runtime.png
manuscript/initial_results_bace.md
```

## Project Structure

```text
quantum-chemistry-kernel/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── manuscript/
├── notebooks/
├── results/
│   ├── figures/
│   ├── kernels/
│   └── tables/
├── scripts/
└── src/
    ├── data/
    ├── evaluation/
    ├── features/
    ├── models/
    └── utils/
```

## Research Framing

This project is a pilot feasibility study of hybrid quantum-classical molecular classification. It demonstrates that small quantum kernel models can be combined with chemistry-aware molecular features and classical PCA compression in a reproducible CPU-only workflow.

The results should not be interpreted as evidence of quantum advantage. Instead, they provide a practical baseline for future experiments involving larger datasets, cross-validation, learnable quantum feature maps, and additional molecular property prediction tasks.
