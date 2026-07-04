import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def smiles_to_mol(smiles: str):
    """Convert SMILES string to RDKit molecule."""
    return Chem.MolFromSmiles(smiles)


def compute_ecfp4(mol, n_bits: int = 1024):
    """
    Compute ECFP4/Morgan fingerprint.
    Radius 2 corresponds to ECFP4.
    """
    generator = GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def compute_descriptors(mol):
    """Compute simple physicochemical molecular descriptors."""
    return np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumRotatableBonds(mol),
            Descriptors.HeavyAtomCount(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ],
        dtype=np.float32,
    )


def build_feature_matrix(df: pd.DataFrame, n_bits: int = 1024):
    """
    Convert dataframe with columns [smiles, label] into molecular feature matrix.

    Output:
    - X: ECFP4 + descriptors
    - y: labels
    - valid_df: dataframe after removing invalid SMILES
    """
    features = []
    labels = []
    valid_rows = []

    for _, row in df.iterrows():
        mol = smiles_to_mol(row["smiles"])

        if mol is None:
            continue

        ecfp = compute_ecfp4(mol, n_bits=n_bits)
        desc = compute_descriptors(mol)

        x = np.concatenate([ecfp, desc])
        features.append(x)
        labels.append(int(row["label"]))
        valid_rows.append(row)

    X = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    return X, y, valid_df
