import pandas as pd
from pathlib import Path


def load_bace(csv_path: str) -> pd.DataFrame:
    """
    Load BACE dataset.

    Expected columns:
    - mol or smiles: SMILES string
    - Class: binary label 0/1
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "mol" in df.columns:
        df = df.rename(columns={"mol": "smiles"})
    elif "SMILES" in df.columns:
        df = df.rename(columns={"SMILES": "smiles"})

    if "Class" not in df.columns:
        raise ValueError("BACE dataset must contain a 'Class' column.")

    df = df[["smiles", "Class"]].dropna()
    df = df.rename(columns={"Class": "label"})

    return df
