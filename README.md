# Quantum Chemistry Kernel

This project implements a lightweight hybrid quantum-classical pipeline for molecular classification using chemistry-aware molecular features and small quantum kernels.

The initial experiment focuses on the BACE classification dataset and evaluates whether a small quantum kernel model can be practically integrated into a reproducible cheminformatics workflow on CPU-only laptop hardware.

## Current Pipeline

```text
SMILES molecules
→ ECFP4 fingerprint + physicochemical descriptors
→ Standardization
→ PCA compression
→ Classical baselines and quantum kernel models
→ Performance, calibration, and runtime evaluation
