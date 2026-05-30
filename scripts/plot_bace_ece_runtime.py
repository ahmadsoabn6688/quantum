from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = Path("results/tables/bace_combined_subset_results.csv")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_PATH)

    label_map = {
        "RBF-SVM-subset": "RBF-SVM",
        "RandomForest-subset": "RF",
        "XGBoost-subset": "XGBoost",
        "QSVM-fixed": "QSVM-full",
        "QAnchor-SVM-fixed": "QAnchor",
    }

    labels = []
    for _, row in df.iterrows():
        name = label_map.get(row["model"], row["model"])
        if name == "QAnchor":
            name = f"QAnchor-{row['n_anchors']}"
        labels.append(name)

    plt.figure(figsize=(8, 5))
    plt.scatter(df["runtime_sec"], df["ece"], s=80)

    for x, y, label in zip(df["runtime_sec"], df["ece"], labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

    plt.xscale("log")
    plt.xlabel("Runtime in seconds, log scale")
    plt.ylabel("Expected Calibration Error, lower is better")
    plt.title("BACE Classification: Calibration Error vs Runtime")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIG_DIR / "bace_ece_vs_runtime.png"
    plt.savefig(out_path, dpi=300)

    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
