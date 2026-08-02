# Independent external benchmarking of freely available *in silico* ADMET prediction tools under chemical-novelty stratification

Data and code accompanying the manuscript. Everything needed to reproduce the
curated test sets, the novelty stratification, and every reported metric is
included here.

**Tools benchmarked:** SwissADME, pkCSM, ADMETlab 3.0, ADMET-AI
**Endpoints:** aqueous solubility, hERG blockade, blood–brain-barrier penetration, drug-induced liver injury

---

## Repository layout

```
admet-bench/
├── code/
│   ├── standardize.py     structure standardization + novelty stratification
│   └── evaluate.py        metrics, bootstrap CIs, DeLong test
├── data/
│   ├── ground_truth/      curated test sets (SMILES, labels, novelty bins)
│   ├── reference/         DrugBank approved-drug set used as the novelty anchor
│   └── predictions/       raw, unmodified output files from each tool
└── results/
    ├── *_scores.csv                per-endpoint metrics (overall + per bin)
    ├── delong_pairwise.csv         pairwise AUROC significance tests
    ├── per_compound_predictions.csv  every compound × every tool
    └── figure1_synthesis.png       manuscript Figure 1
```

## Installation

```bash
pip install rdkit pandas numpy scikit-learn scipy
```

Versions used: RDKit 2026.03, pandas 2.x, scikit-learn 1.x, SciPy 1.x, Python 3.12.

---

## Datasets

| Endpoint | n | Label definition | Original source |
|---|---|---|---|
| Solubility | 132 | intrinsic logS₀ (log mol/L), regression | ACS Second Solubility Challenge (SC-2) |
| hERG | 84 | blocker at 10 µM IC₅₀ (41 + / 43 −) | CardioTox external Test Sets I + III |
| BBB | 184 | penetrant at logBB > −1 (92 + / 92 −) | B3DB regression subset, BBBP/Martins overlap removed |
| DILI | 198 | vMost+vLess = +, vNo = − (99 + / 99 −) | FDA DILIrank / DILIst curated gold standard |

Each ground-truth file contains the standardized SMILES, InChIKey, experimental
value or label, nearest-neighbour Tanimoto similarity, novelty bin, and source.

### Novelty stratification

Novelty is the maximum ECFP4 (Morgan radius 2, 2048-bit) Tanimoto similarity to
a reference set of approved small-molecule drugs:

| Bin | Range |
|---|---|
| Near | Tc ≥ 0.6 |
| Intermediate | 0.4 ≤ Tc < 0.6 |
| Novel | Tc < 0.4 |

The 0.4 boundary follows the train/test leakage threshold used in low-similarity
benchmark studies (Steshin, *Lo-Hi*, NeurIPS 2023).

The reference set (`data/reference/drugbank_approved_reference.csv`) contains
**1,550 unique** approved small molecules, derived from the openly available
DrugBank "slim" approved-drug list (1,552 records parsed, of which two were
InChIKey duplicates; because novelty is a **maximum** similarity, duplicate
reference entries cannot change any result).

---

## Reproducing the analysis

### 1. Standardization and novelty bins

```bash
cd code
python standardize.py \
    --input ../data/ground_truth/herg_cardiotox_groundtruth.csv \
    --smiles-col smiles \
    --reference ../data/reference/drugbank_approved_reference.csv \
    --output /tmp/herg_annotated.csv
```

Expected output:

```
input compounds: 84
parsed + standardized: 84
after InChIKey deduplication: 84 (0 removed)
reference set fingerprints: 1550
novelty bins: {'Near': 35, 'Intermediate': 10, 'Novel': 39}
```

This reproduces the published bins exactly (verified: zero Tanimoto difference,
zero bin reassignments across all 84 compounds).

To reproduce the BBB leakage removal, pass a training set:

```bash
python standardize.py --input <b3db_regression.csv> --smiles-col SMILES \
    --reference ../data/reference/drugbank_approved_reference.csv \
    --remove-training <BBBP.csv> --training-smiles-col smiles \
    --output /tmp/bbb_annotated.csv
```

### 2. Metrics and significance tests

```bash
python evaluate.py \
    --predictions ../results/per_compound_predictions.csv \
    --endpoint "hERG (blocker@10uM)" \
    --tool-cols ADMETlab3.0 ADMET-AI \
    --delong ADMETlab3.0 ADMET-AI
```

Expected: ADMETlab 3.0 AUROC 0.840, ADMET-AI AUROC 0.748, DeLong p ≈ 0.085.

Endpoint strings are `Solubility (logS0)`, `hERG (blocker@10uM)`,
`BBB (logBB>-1)`, `DILI`. Use `--kind regression` for solubility.

---

## Notes on the raw prediction files

`data/predictions/` holds the tool outputs exactly as downloaded, unmodified.
Two quirks are worth knowing when re-parsing them:

- **pkCSM** files are tab-separated with each row wrapped in double quotes; read
  with `sep='\t', quoting=csv.QUOTE_NONE` and strip quotes from the header and
  SMILES column.
- **pkCSM** consistently fails to return the first compound of an uploaded
  batch. In this study a disposable structure was prepended to each upload and
  its output discarded; some files are therefore split into batches.
- **SwissADME** rejects SMILES longer than 200 characters, so two large
  molecules in the BBB set have no SwissADME prediction. The BBB files are
  split into three batches for this reason.

Predictions are matched to ground truth by **InChIKey connectivity block**, not
row order, so dropped or reordered compounds are detected automatically.

## Prediction column semantics

| Tool | Output type |
|---|---|
| SwissADME | ESOL logS (solubility, log mol/L); BOILED-Egg Yes/No (BBB) |
| pkCSM | numeric log mol/L (solubility); numeric logBB (BBB); Yes/No (hERG II, hepatotoxicity) |
| ADMETlab 3.0 | numeric logS; probability [0–1] for classification endpoints |
| ADMET-AI | numeric logS; probability [0–1] for classification endpoints |

Binary-only outputs yield MCC, sensitivity and specificity but not AUROC/AUPRC.

## Excluded tool

A fifth, in-house platform (RADPT) was assessed during method development but
excluded from the quantitative comparison: its per-molecule runtime and cost and
its non-tabular ordinal output were incompatible with the automated batch
protocol. No RADPT results are reported in the manuscript.

## License

Code released under the MIT License. Ground-truth datasets remain subject to the
terms of their original sources (SC-2, CardioTox, B3DB, DILIrank/DILIst,
DrugBank); please cite those primary sources alongside this repository.

## Citation

> [Sagar agrawal, Shankar Gejji, Naveen Maurya, Rakesh Kumar Dixit, Rahul Kumar ]. Independent external benchmarking of freely available *in silico*
> ADMET prediction tools under chemical-novelty stratification. [Journal], [2026].
