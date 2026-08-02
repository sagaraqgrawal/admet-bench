"""
standardize.py — structure standardization and chemical-novelty stratification.

This is the exact pipeline used to prepare every test compound in the study.
Because ECFP4 fingerprints (and therefore the novelty bins) depend on how
structures are standardized, this script is provided so the novelty axis is
fully reproducible.

Pipeline
--------
1. Parse SMILES with RDKit.
2. Keep the largest organic fragment (removes salts / solvates / counterions).
3. Normalize functional groups (RDKit MolStandardize Normalizer).
4. Neutralize charges (Uncharger).
5. Emit canonical SMILES + standard InChIKey.
6. Deduplicate on the first InChIKey block (connectivity layer; ignores
   stereochemistry and isotopes).
7. Compute max ECFP4 (Morgan r=2, 2048-bit) Tanimoto similarity to a reference
   set of approved small-molecule drugs, and assign a novelty bin.

Novelty bins
------------
    Near          Tc >= 0.6
    Intermediate  0.4 <= Tc < 0.6
    Novel         Tc <  0.4

The 0.4 boundary matches the train/test leakage threshold used in
low-similarity benchmark studies (Steshin, Lo-Hi, NeurIPS 2023).

Usage
-----
    python standardize.py --input compounds.csv --smiles-col SMILES \
        --reference ../data/reference/drugbank_approved_reference.csv \
        --output annotated.csv

Requirements: rdkit, pandas, numpy
"""
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Novelty bin edges (see module docstring)
NEAR_MIN = 0.6
NOVEL_MAX = 0.4

# ECFP4 = Morgan fingerprint, radius 2, 2048 bits
FP_RADIUS = 2
FP_BITS = 2048

_normalizer = rdMolStandardize.Normalizer()
_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_mol(smiles):
    """Return a standardized RDKit Mol, or None if the SMILES cannot be parsed."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = _largest_fragment.choose(mol)   # desalt / strip solvates
    mol = _normalizer.normalize(mol)      # normalize functional groups
    mol = _uncharger.uncharge(mol)        # neutralize
    return mol


def standardize_frame(df, smiles_col):
    """Standardize a dataframe of structures; adds SMILES_std / InChIKey / ik14."""
    out = []
    for _, row in df.iterrows():
        mol = standardize_mol(row[smiles_col])
        if mol is None:
            continue
        rec = row.to_dict()
        inchikey = Chem.MolToInchiKey(mol)
        rec["SMILES_std"] = Chem.MolToSmiles(mol)
        rec["InChIKey"] = inchikey
        rec["ik14"] = inchikey.split("-")[0]   # connectivity layer
        out.append(rec)
    return pd.DataFrame(out)


def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, FP_BITS)


def load_reference_fps(reference_csv, smiles_col="SMILES"):
    """Fingerprint the approved-drug reference set used for novelty scoring."""
    ref = pd.read_csv(reference_csv)
    fps = []
    for smi in ref[smiles_col]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(morgan_fp(mol))
    return fps


def nearest_neighbour_tanimoto(smiles_std, reference_fps):
    """Max Tanimoto similarity of one compound to the reference set."""
    mol = Chem.MolFromSmiles(str(smiles_std))
    if mol is None:
        return np.nan
    sims = DataStructs.BulkTanimotoSimilarity(morgan_fp(mol), reference_fps)
    return max(sims) if sims else np.nan


def novelty_bin(tc):
    if pd.isna(tc):
        return None
    if tc < NOVEL_MAX:
        return "Novel"
    if tc < NEAR_MIN:
        return "Intermediate"
    return "Near"


def annotate_novelty(df, reference_fps, smiles_col="SMILES_std"):
    tc = [nearest_neighbour_tanimoto(s, reference_fps) for s in df[smiles_col]]
    df = df.copy()
    df["nn_tanimoto"] = np.round(tc, 3)
    df["novelty_bin"] = [novelty_bin(t) for t in tc]
    return df


def remove_training_overlap(df, training_csv, training_smiles_col="smiles"):
    """
    Drop test compounds that appear in a known training set, matched on the
    InChIKey connectivity block. Used for BBB (BBBP/Martins removal).
    Returns (filtered_df, n_removed).
    """
    train = pd.read_csv(training_csv)
    keys = set()
    for smi in train[training_smiles_col]:
        mol = standardize_mol(smi)
        if mol is not None:
            keys.add(Chem.MolToInchiKey(mol).split("-")[0])
    mask = ~df["ik14"].isin(keys)
    return df[mask].copy(), int((~mask).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV of compounds to process")
    ap.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    ap.add_argument("--reference", required=True,
                    help="approved-drug reference CSV (novelty anchor)")
    ap.add_argument("--remove-training", default=None,
                    help="optional training-set CSV whose compounds are removed")
    ap.add_argument("--training-smiles-col", default="smiles")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"input compounds: {len(df)}")

    df = standardize_frame(df, args.smiles_col)
    print(f"parsed + standardized: {len(df)}")

    before = len(df)
    df = df.drop_duplicates("ik14")
    print(f"after InChIKey deduplication: {len(df)} ({before - len(df)} removed)")

    if args.remove_training:
        df, n_removed = remove_training_overlap(
            df, args.remove_training, args.training_smiles_col)
        print(f"training-set overlap removed: {n_removed} -> {len(df)} remain")

    ref_fps = load_reference_fps(args.reference)
    print(f"reference set fingerprints: {len(ref_fps)}")

    df = annotate_novelty(df, ref_fps)
    counts = df["novelty_bin"].value_counts().reindex(
        ["Near", "Intermediate", "Novel"]).fillna(0).astype(int)
    print("novelty bins:", counts.to_dict())

    df.to_csv(args.output, index=False)
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
