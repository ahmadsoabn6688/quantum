from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


SEED = 0
TEST_SIZE = 0.15

IN_DIR = Path("data/processed")
OUT_DIR = Path("data/processed")


def make_pca_version(X_train, X_test, n_components):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=SEED)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, scaler, pca


def main():
    X = np.load(IN_DIR / "bace_X.npy")
    y = np.load(IN_DIR / "bace_y.npy")
    df = pd.read_csv(IN_DIR / "bace_valid.csv")

    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    np.save(OUT_DIR / "bace_train_idx.npy", train_idx)
    np.save(OUT_DIR / "bace_test_idx.npy", test_idx)
    np.save(OUT_DIR / "bace_y_train.npy", y_train)
    np.save(OUT_DIR / "bace_y_test.npy", y_test)

    df.iloc[train_idx].to_csv(OUT_DIR / "bace_train.csv", index=False)
    df.iloc[test_idx].to_csv(OUT_DIR / "bace_test.csv", index=False)

    for n_components in [4, 6]:
        X_train_pca, X_test_pca, scaler, pca = make_pca_version(
            X_train, X_test, n_components
        )

        np.save(OUT_DIR / f"bace_X_train_pca{n_components}.npy", X_train_pca)
        np.save(OUT_DIR / f"bace_X_test_pca{n_components}.npy", X_test_pca)

        joblib.dump(scaler, OUT_DIR / f"bace_scaler_pca{n_components}.joblib")
        joblib.dump(pca, OUT_DIR / f"bace_pca{n_components}.joblib")

        print(f"PCA {n_components}")
        print(f"Train shape: {X_train_pca.shape}")
        print(f"Test shape: {X_test_pca.shape}")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
