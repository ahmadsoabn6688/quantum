from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.evaluation.calibration import expected_calibration_error


DATA_DIR = Path("data/processed")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
PCA_DIM = 4
N_TRAIN_SUBSET = 300
N_TEST_SUBSET = 100


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    start = time.time()

    model.fit(X_train, y_train)
    runtime = time.time() - start

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    result = {
        "model": name,
        "pca_dim": PCA_DIM,
        "n_train": N_TRAIN_SUBSET,
        "n_test": N_TEST_SUBSET,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auroc": roc_auc_score(y_test, y_score),
        "auprc": average_precision_score(y_test, y_score),
        "ece": expected_calibration_error(y_test, y_score, n_bins=10),
        "runtime_sec": runtime,
    }

    return result


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

    models = [
        (
            "RBF-SVM-subset",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=SEED,
            ),
        ),
        (
            "RandomForest-subset",
            RandomForestClassifier(
                n_estimators=200,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost-subset",
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
    ]

    results = []

    for name, model in models:
        print(f"Training {name}...")
        result = evaluate_model(
            name,
            model,
            X_train_small,
            y_train_small,
            X_test_small,
            y_test_small,
        )
        results.append(result)
        print(result)

    results_df = pd.DataFrame(results)
    out_path = OUT_DIR / "bace_subset_baseline_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\nSubset baseline results:")
    print(results_df)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
