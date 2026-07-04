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


DATA_DIR = Path("data/processed")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0


def evaluate_model(name, model, X_train, y_train, X_test, y_test, pca_dim):
    start = time.time()

    model.fit(X_train, y_train)

    runtime = time.time() - start

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    results = {
        "model": name,
        "pca_dim": pca_dim,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auroc": roc_auc_score(y_test, y_score),
        "auprc": average_precision_score(y_test, y_score),
        "runtime_sec": runtime,
    }

    return results


def main():
    all_results = []

    y_train = np.load(DATA_DIR / "bace_y_train.npy")
    y_test = np.load(DATA_DIR / "bace_y_test.npy")

    for pca_dim in [4, 6]:
        print(f"\nRunning baselines with PCA-{pca_dim}")

        X_train = np.load(DATA_DIR / f"bace_X_train_pca{pca_dim}.npy")
        X_test = np.load(DATA_DIR / f"bace_X_test_pca{pca_dim}.npy")

        models = [
            (
                "RBF-SVM",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=SEED,
                ),
            ),
            (
                "RandomForest",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
            (
                "XGBoost",
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

        for name, model in models:
            print(f"Training {name}...")
            result = evaluate_model(
                name, model, X_train, y_train, X_test, y_test, pca_dim
            )
            all_results.append(result)
            print(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "bace_baseline_results.csv", index=False)

    print("\nFinal results:")
    print(results_df)
    print(f"\nSaved to: {OUT_DIR / 'bace_baseline_results.csv'}")


if __name__ == "__main__":
    main()
