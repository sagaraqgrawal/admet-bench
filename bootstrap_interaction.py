"""Does the novelty effect differ BETWEEN endpoints?

The per-tool Near-vs-Novel bootstrap tests "does this tool degrade on this
endpoint", which is not the manuscript's claim. The claim is comparative:
degradation is confined to DILI and absent from the structurally grounded
endpoints. That is an interaction, and it is tested here directly as a
difference of differences.

Two statistics:

1. Pooled within-endpoint degradation. For an endpoint, both probabilistic
   tools are scored on the SAME resampled compounds and their AUROCs averaged
   before the Near - Novel difference is taken. This respects the fact that
   the two tools see identical molecules, which the per-tool test ignores, and
   it is the quantity the manuscript actually argues about ("the direction is
   consistent across tools").

2. The interaction. delta(DILI) - delta(other endpoint), with the two
   endpoints resampled independently because they are disjoint compound sets.
   Positive means DILI degrades more with novelty than that endpoint does.

Same stratified resampling as bootstrap_novelty.py: classes resampled
separately within each stratum, 10 000 replicates.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from pathlib import Path

# Supplementary Table S1 — the per-compound workbook this analysis reads.
# Looked for under data/ and then beside this script, so the script works
# whether it is run from the repository root or from its own directory.
SHEET = "Per_Compound_Data"

# The per-compound workbook this analysis reads. Searched for in the
# locations it occupies in this repository, so the script runs from the
# repository root or from its own directory.
HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    HERE / "results" / "Supplementary_Results.xlsx",
    HERE.parent / "results" / "Supplementary_Results.xlsx",
    HERE / "data" / "Supplementary_Information.xlsx",
    HERE / "Supplementary_Information.xlsx",
]
XLSX = next((c for c in _CANDIDATES if c.exists()), None)
if XLSX is None:
    raise SystemExit(
        "Per-compound workbook not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in _CANDIDATES)
        + "\nEdit the _CANDIDATES list to point at its location."
    )

_sheets = pd.ExcelFile(XLSX).sheet_names
if SHEET not in _sheets:
    raise SystemExit(
        f"{XLSX.name} has no sheet named {SHEET!r}. Sheets present: "
        + ", ".join(_sheets)
    )
N_BOOT = 10000
SEED = 20261104

num = lambda s: pd.to_numeric(s, errors="coerce")

# endpoints with >= 2 probabilistic tools, so a pooled score is meaningful
ENDPOINTS = {
    "hERG": ("hERG (blocker@10uM)", ["ADMETlab3.0_score", "ADMET-AI_score"]),
    "BBB": ("BBB (logBB>-1)", ["ADMET-AI_score", "ADMETlab3.0_score", "pkCSM_score"]),
    "DILI": ("DILI", ["ADMETlab3.0_score", "ADMET-AI_score"]),
}

df = pd.read_excel(XLSX, sheet_name=SHEET)
df = df.dropna(subset=["endpoint"])


def auc_rows(pos, neg):
    n_pos, n_neg = pos.shape[1], neg.shape[1]
    r = rankdata(np.concatenate([pos, neg], axis=1), axis=1)
    return (r[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (
        n_pos * n_neg)


def pooled_draws(sub, cols, bin_name, rng, n_boot=N_BOOT):
    """Mean AUROC across tools, tools scored on identical resampled rows."""
    s = sub[sub["novelty_bin"] == bin_name][["ground_truth"] + cols].copy()
    for c in s.columns:
        s[c] = num(s[c])
    s = s.dropna()
    y = s["ground_truth"].values
    mats = {c: s[c].values for c in cols}
    ipos, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    pi = rng.integers(0, len(ipos), size=(n_boot, len(ipos)))
    ni = rng.integers(0, len(ineg), size=(n_boot, len(ineg)))
    per_tool = []
    for c in cols:
        p = mats[c]
        per_tool.append(auc_rows(p[ipos][pi], p[ineg][ni]))
    return np.mean(per_tool, axis=0), len(y)


def pooled_observed(sub, cols, bin_name):
    s = sub[sub["novelty_bin"] == bin_name][["ground_truth"] + cols].copy()
    for c in s.columns:
        s[c] = num(s[c])
    s = s.dropna()
    y = s["ground_truth"].values
    return float(np.mean([roc_auc_score(y, s[c].values) for c in cols]))


def summarize(d, obs, label):
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    pv = max(min(pv, 1.0), 1.0 / len(d))
    star = "  <-- p<0.05" if pv < 0.05 else ""
    print(f"  {label:46s} d={obs:+6.3f}  "
          f"CI[{lo:+6.3f},{hi:+6.3f}]  p={pv:.4f}{star}")
    return dict(label=label, delta=round(float(obs), 3),
                ci_low=round(float(lo), 3), ci_high=round(float(hi), 3),
                p=round(float(pv), 4))


master = np.random.default_rng(SEED)
deltas, rows = {}, []

print("1. POOLED Near - Novel degradation, per endpoint")
print("   (both/all probabilistic tools on identical resampled compounds;")
print("    positive = worse on novel chemistry)\n")

for name, (ep, cols) in ENDPOINTS.items():
    sub = df[df["endpoint"] == ep]
    rng = np.random.default_rng(master.integers(2 ** 32))
    near, n_near = pooled_draws(sub, cols, "Near", rng)
    rng = np.random.default_rng(master.integers(2 ** 32))
    novel, n_novel = pooled_draws(sub, cols, "Novel", rng)
    d = near - novel
    obs = pooled_observed(sub, cols, "Near") - pooled_observed(
        sub, cols, "Novel")
    deltas[name] = d
    rows.append(summarize(
        d, obs, f"{name} (n={n_near}/{n_novel}, {len(cols)} tools)"))

print("\n2. INTERACTION: is the DILI degradation larger than the others?")
print("   (positive = DILI loses more accuracy with novelty)\n")

for other in ["hERG", "BBB"]:
    d = deltas["DILI"] - deltas[other]
    obs = float(np.mean(deltas["DILI"]) - np.mean(deltas[other]))
    rows.append(summarize(d, obs, f"DILI minus {other}"))

both = (deltas["hERG"] + deltas["BBB"]) / 2.0
d = deltas["DILI"] - both
rows.append(summarize(
    d, float(np.mean(deltas["DILI"]) - np.mean(both)),
    "DILI minus mean(hERG, BBB)"))

OUT = HERE / "bootstrap_interaction_results.csv"
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
