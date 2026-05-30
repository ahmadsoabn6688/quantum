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


def scale_to_angles(X):
    """
    Scale PCA features to a small angle range for quantum encoding.
    This keeps circuit rotations numerically stable.
    """
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

    # Important: scale train and test together for this first simple experiment
    X_all = np.vstack([X_train_small, X_test_small])
    X_all_angles = scale_to_angles(X_all)

    X_train_angles = X_all_angles[:N_TRAIN_SUBSET]
    X_test_angles = X_all_angles[N_TRAIN_SUBSET:]

    print("Running fixed quantum kernel experiment")
    print(f"PCA dim / qubits: {PCA_DIM}")
    print(f"Train subset: {X_train_angles.shape}")
    print(f"Test subset: {X_test_angles.shape}")

    qkernel = QuantumKernel(n_qubits=PCA_DIM, depth=1)

    start_kernel = time.time()

    print("Computing train quantum kernel...")
    K_train = qkernel.kernel_matrix(X_train_angles)

    print("Computing test quantum kernel...")
    K_test = qkernel.kernel_matrix(X_test_angles, X_train_angles)

    kernel_time = time.time() - start_kernel

    np.save(KERNEL_DIR / "bace_qkernel_train_pca4_n300.npy", K_train)
    np.save(KERNEL_DIR / "bace_qkernel_test_pca4_n100.npy", K_test)

    print("Training QSVM...")
    start_train = time.time()

    model = SVC(kernel="precomputed", probability=True, random_state=SEED)
    model.fit(K_train, y_train_small)

    train_time = time.time() - start_train

    y_pred = model.predict(K_test)
    y_score = model.predict_proba(K_test)[:, 1]

    results = {
        "model": "QSVM-fixed",
        "pca_dim": PCA_DIM,
        "n_train": N_TRAIN_SUBSET,
        "n_test": N_TEST_SUBSET,
        "accuracy": accuracy_score(y_test_small, y_pred),
        "f1": f1_score(y_test_small, y_pred),
        "auroc": roc_auc_score(y_test_small, y_score),
	"ece": expected_calibration_error(y_test_small, y_score, n_bins=10),
        "auprc": average_precision_score(y_test_small, y_score),
        "kernel_time_sec": kernel_time,
        "svm_train_time_sec": train_time,
        "total_runtime_sec": kernel_time + train_time,
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(OUT_DIR / "bace_quantum_kernel_results.csv", index=False)

    print("\nQuantum result:")
    print(results_df)
    print(f"\nSaved to: {OUT_DIR / 'bace_quantum_kernel_results.csv'}")


if __name__ == "__main__":
    main()

