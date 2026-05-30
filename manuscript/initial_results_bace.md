# Initial BACE Results

We conducted an initial CPU-only laptop-scale experiment on the BACE molecular classification dataset. Each molecule was represented using a chemistry-aware molecular feature vector composed of a 1024-bit ECFP4 fingerprint and eight physicochemical descriptors. The resulting 1032-dimensional representation was standardized and compressed using PCA to obtain low-dimensional latent molecular representations suitable for quantum encoding.

The experiment used a fixed subset with 300 training molecules and 100 test molecules. Classical baselines included RBF-SVM, Random Forest, and XGBoost. Quantum models included a fixed four-qubit quantum kernel SVM, anchor-based four-qubit quantum approximations with 64 and 128 anchors, and a six-qubit anchor-based quantum approximation with 64 anchors.

The full four-qubit QSVM achieved an accuracy of 0.700, F1 score of 0.667, AUROC of 0.714, AUPRC of 0.699, and ECE of 0.092. This was comparable to the RBF-SVM baseline, which achieved an accuracy of 0.690, F1 score of 0.667, AUROC of 0.708, AUPRC of 0.682, and ECE of 0.080 on the same subset.

The 128-anchor four-qubit approximation preserved most of the full QSVM performance, achieving an accuracy of 0.700, AUROC of 0.708, AUPRC of 0.700, and ECE of 0.074, while reducing runtime from 280.3 seconds to 185.9 seconds. The 64-anchor four-qubit approximation further reduced runtime to 93.2 seconds, but with lower predictive performance.

The six-qubit anchor-based quantum model achieved the best quantum ranking performance, with an AUROC of 0.752 and AUPRC of 0.743. This suggests that increasing the quantum feature dimension from four to six qubits can improve ranking-based molecular classification performance. However, this improvement came with higher runtime and a higher ECE compared with the four-qubit anchor models.

Overall, the Random Forest baseline remained the strongest classical model, achieving the best AUROC and lowest ECE. However, the quantum kernel models demonstrated that small quantum feature maps can be integrated into a reproducible chemistry-aware machine-learning pipeline using only CPU-based laptop hardware. The aim of this experiment is not to claim quantum advantage, but to evaluate the practical feasibility of combining molecular fingerprints, classical dimensionality reduction, and small quantum kernels for molecular classification.
