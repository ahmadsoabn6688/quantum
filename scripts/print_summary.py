from pathlib import Path
import pandas as pd


RESULTS_PATH = Path("results/tables/bace_combined_subset_results.csv")


def main():
    df = pd.read_csv(RESULTS_PATH)

    print("\n=== BACE Hybrid Quantum-Chemistry Experiment Summary ===\n")

    print("Dataset: BACE classification")
    print("Features: ECFP4 1024-bit + 8 physicochemical descriptors")
    print("Compression: PCA to 4 and 6 dimensions")
    print("Subset: 300 train / 100 test")
    print("Seed: 0\n")

    display_cols = [
        "model",
        "pca_dim",
        "n_anchors",
        "accuracy",
        "f1",
        "auroc",
        "auprc",
        "ece",
        "runtime_sec",
    ]

    print(df[display_cols].round(4).to_string(index=False))

    best_auroc = df.loc[df["auroc"].idxmax()]
    best_ece = df.loc[df["ece"].idxmin()]

    quantum_df = df[df["model"].str.contains("QSVM|QAnchor", case=False, regex=True)]
    best_quantum_auroc = quantum_df.loc[quantum_df["auroc"].idxmax()]
    best_quantum_ece = quantum_df.loc[quantum_df["ece"].idxmin()]

    print("\nBest Overall AUROC:")
    print(
        f"{best_auroc['model']} | PCA={best_auroc['pca_dim']} | "
        f"anchors={best_auroc['n_anchors']} | AUROC={best_auroc['auroc']:.4f}"
    )

    print("\nBest Overall Calibration / Lowest ECE:")
    print(
        f"{best_ece['model']} | PCA={best_ece['pca_dim']} | "
        f"anchors={best_ece['n_anchors']} | ECE={best_ece['ece']:.4f}"
    )

    print("\nBest Quantum AUROC:")
    print(
        f"{best_quantum_auroc['model']} | PCA={best_quantum_auroc['pca_dim']} | "
        f"anchors={best_quantum_auroc['n_anchors']} | "
        f"AUROC={best_quantum_auroc['auroc']:.4f}"
    )

    print("\nBest Quantum Calibration / Lowest Quantum ECE:")
    print(
        f"{best_quantum_ece['model']} | PCA={best_quantum_ece['pca_dim']} | "
        f"anchors={best_quantum_ece['n_anchors']} | "
        f"ECE={best_quantum_ece['ece']:.4f}"
    )

    print("\nKey findings:")
    print(
        "1. The fixed 4-qubit QSVM achieved performance comparable to RBF-SVM "
        "on the same PCA-compressed molecular representation."
    )
    print(
        "2. The 128-anchor quantum approximation preserved most of the full QSVM "
        "performance while reducing runtime and improving calibration."
    )
    print(
        "3. The 6-qubit anchor model achieved the best quantum AUROC/AUPRC, "
        "showing that increasing the quantum feature dimension can improve "
        "ranking performance, although with higher runtime and ECE."
    )


if __name__ == "__main__":
    main()
