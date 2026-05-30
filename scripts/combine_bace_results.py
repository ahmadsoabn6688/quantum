from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results/tables")
OUT_PATH = RESULTS_DIR / "bace_combined_subset_results.csv"


def load_result(filename, anchors=None):
    df = pd.read_csv(RESULTS_DIR / filename)

    if "total_runtime_sec" in df.columns:
        df = df.rename(columns={"total_runtime_sec": "runtime_sec"})

    if anchors is not None:
        df["n_anchors"] = anchors

    return df


def main():
    dfs = []

    baseline = load_result("bace_subset_baseline_results.csv")
    baseline["n_anchors"] = "-"
    dfs.append(baseline)

    dfs.append(load_result("bace_quantum_kernel_results.csv", anchors="full"))
    dfs.append(load_result("bace_quantum_anchor_results.csv", anchors=64))
    dfs.append(load_result("bace_quantum_anchor_results_a128.csv", anchors=128))
    dfs.append(load_result("bace_quantum_anchor_results_6q_a64.csv", anchors=64))

    common_cols = [
        "model",
        "pca_dim",
        "n_train",
        "n_test",
        "n_anchors",
        "accuracy",
        "f1",
        "auroc",
        "auprc",
        "ece",
        "runtime_sec",
    ]

    combined = pd.concat([df[common_cols] for df in dfs], ignore_index=True)

    combined.to_csv(OUT_PATH, index=False)

    print("\nCombined BACE subset results:")
    print(combined.round(4))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
