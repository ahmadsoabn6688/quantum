from src.evaluation.calibration import expected_calibration_error
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.models.quantum_kernel import QuantumKernel


DATA_DIR = Path("data/processed")
OUT_DIR = Path("results/tables")
KERNEL_DIR = Path("results/kernels")

OUT_DIR.mkdir(parents=True, exist_ok=True)
KERNEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
PCA_DIM = 4
N_TRAIN_SUBSET = 300
N_TEST_SUBSET = 100
N_ANCHORS = 64


def scale_to_angles(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
    return X_scaled * np.pi


def main():
    rng = np.random.default_rng(SEED)

    X_train = np.load(DATA_DIR / f"bace_X_train_pca{PCA_DIM}.npy")
    X_test = np.load(DATA_DIR / f"bace_X_test_pca{PCA_DIM}.npy")
    y_train = np.load(DATA_DIR / "bace_y_train.npy")
    y_test = np.load(DATA_DIR / "bace_y_test.npy")

    train_idx = rng.choice(len(X_train), size=N_TRAIN_SUBSET, replace=False)
    test_idx = rng.choice(len(X_test), size=N_TEST_SUBSET, replace=False)

    X_train_small = X_train[train_idx]
    y_train_small = y_train[train_idx]
    X_test_small = X_test[test_idx]
    y_test_small = y_test[test_idx]

    X_all = np.vstack([X_train_small, X_test_small])
    X_all_angles = scale_to_angles(X_all)

    X_train_angles = X_all_angles[:N_TRAIN_SUBSET]
    X_test_angles = X_all_angles[N_TRAIN_SUBSET:]

    anchor_idx = rng.choice(N_TRAIN_SUBSET, size=N_ANCHORS, replace=False)
    X_anchor_angles = X_train_angles[anchor_idx]

    np.save(KERNEL_DIR / "bace_anchor_indices_pca4_n64.npy", anchor_idx)

    print("Running anchor-based quantum kernel experiment")
    print(f"PCA dim / qubits: {PCA_DIM}")
    print(f"Train subset: {X_train_angles.shape}")
    print(f"Test subset: {X_test_angles.shape}")
    print(f"Anchors: {X_anchor_angles.shape}")

    qkernel = QuantumKernel(n_qubits=PCA_DIM, depth=1)

    start_kernel = time.time()

    print("Computing train-to-anchor quantum features...")
    Z_train = qkernel.kernel_matrix(X_train_angles, X_anchor_angles)

    print("Computing test-to-anchor quantum features...")
    Z_test = qkernel.kernel_matrix(X_test_angles, X_anchor_angles)

    kernel_time = time.time() - start_kernel

    np.save(KERNEL_DIR / "bace_qanchor_train_pca4_n300_a64.npy", Z_train)
    np.save(KERNEL_DIR / "bace_qanchor_test_pca4_n100_a64.npy", Z_test)

    print("Training linear SVM on quantum anchor features...")
    start_train = time.time()

    model = SVC(kernel="linear", probability=True, random_state=SEED)
    model.fit(Z_train, y_train_small)

    train_time = time.time() - start_train

    y_pred = model.predict(Z_test)
    y_score = model.predict_proba(Z_test)[:, 1]

    results = {
        "model": "QAnchor-SVM-fixed",
        "pca_dim": PCA_DIM,
        "n_train": N_TRAIN_SUBSET,
        "n_test": N_TEST_SUBSET,
        "n_anchors": N_ANCHORS,
        "accuracy": accuracy_score(y_test_small, y_pred),
        "f1": f1_score(y_test_small, y_pred),
        "auroc": roc_auc_score(y_test_small, y_score),
        "auprc": average_precision_score(y_test_small, y_score),
	"ece": expected_calibration_error(y_test_small, y_score, n_bins=10),
        "kernel_time_sec": kernel_time,
        "svm_train_time_sec": train_time,
        "total_runtime_sec": kernel_time + train_time,
    }

    results_df = pd.DataFrame([results])
    out_path = OUT_DIR / "bace_quantum_anchor_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\nQuantum anchor result:")
    print(results_df)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
