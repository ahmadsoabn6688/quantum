from pathlib import Path
import numpy as np

from src.data.load_bace import load_bace
from src.features.molecular_features import build_feature_matrix


RAW_PATH = "data/raw/bace.csv"
OUT_DIR = Path("data/processed")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BACE dataset...")
    df = load_bace(RAW_PATH)

    print("Building molecular features...")
    X, y, valid_df = build_feature_matrix(df, n_bits=1024)

    print("Saving processed files...")
    np.save(OUT_DIR / "bace_X.npy", X)
    np.save(OUT_DIR / "bace_y.npy", y)
    valid_df.to_csv(OUT_DIR / "bace_valid.csv", index=False)

    print("Done.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Valid molecules: {len(valid_df)}")


if __name__ == "__main__":
    main()
