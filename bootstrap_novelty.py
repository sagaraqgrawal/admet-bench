"""Significance test for the novelty effect in the ADMET-Bench external data.

Question
--------
Is the difference in performance between novelty strata larger than sampling
noise, and is the degradation specific to DILI?

Method
------
Stratified bootstrap, 10 000 resamples. Within each novelty stratum compounds
are resampled with replacement, resampling each class separately so that a
resample can never lose a class and leave AUROC undefined (this also holds the
stratum's class balance fixed at its observed value). For solubility the whole
stratum is resampled together.

Statistic, per tool and per pair of strata:

    delta = performance(stratum A) - performance(stratum B)

oriented so that POSITIVE always means "better on the more drug-like stratum".
For AUROC that is AUROC(A) - AUROC(B); for RMSE it is RMSE(B) - RMSE(A), since
lower RMSE is better.

Reported: the observed delta, a 95 % percentile-bootstrap interval, and a
two-sided percentile-bootstrap p = 2 x min(P(delta<=0), P(delta>=0)), floored
at 1/n_boot. Note this is a one-sample bootstrap of a difference, not a
permutation test; it answers "is the observed gap distinguishable from zero
given the sample sizes", which is the question a reviewer asks of a
novelty-stratified claim.

AUROC is computed from the Mann-Whitney rank identity rather than by calling
roc_auc_score per replicate, which makes the whole run vectorized. The
identity is exact, including ties (average ranks); agreement with
sklearn.metrics.roc_auc_score is asserted on the observed data before any
resampling.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

XLSX = ("/mnt/user-data/attachments/8bb97ff4-e6fe-5cee-81ac-aaf31743d97f/"
        "fc1aa685-Supplementary_Information.xlsx")
N_BOOT = 10000
SEED = 20261104
BINS = ["Near", "Intermediate", "Novel"]

num = lambda s: pd.to_numeric(s, errors="coerce")

SPEC = {
    "Solubility (logS0)": ("rmse", {"SwissADME": "SwissADME",
                                    "ADMETlab 3.0": "ADMETlab3.0",
                                    "ADMET-AI": "ADMET-AI",
                                    "pkCSM": "pkCSM"}),
    "hERG (blocker@10uM)": ("auroc", {"ADMETlab 3.0": "ADMETlab3.0",
                                      "ADMET-AI": "ADMET-AI"}),
    "BBB (logBB>-1)": ("auroc", {"ADMET-AI": "ADMET-AI",
                                 "pkCSM": "pkCSM",
                                 "ADMETlab 3.0": "ADMETlab3.0"}),
    "DILI": ("auroc", {"ADMETlab 3.0": "ADMETlab3.0",
                       "ADMET-AI": "ADMET-AI"}),
}


def auc_rows(pos, neg):
    """AUROC for each row of (n_boot, n_pos) vs (n_boot, n_neg) score arrays."""
    n_pos, n_neg = pos.shape[1], neg.shape[1]
    allscores = np.concatenate([pos, neg], axis=1)
    r = rankdata(allscores, axis=1)
    return (r[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (
        n_pos * n_neg)


def boot_draws(y, p, kind, rng, n_boot=N_BOOT):
    if kind == "rmse":
        idx = rng.integers(0, len(p), size=(n_boot, len(p)))
        err = p[idx] - y[idx]
        return np.sqrt((err ** 2).mean(axis=1))
    pos, neg = p[y == 1], p[y == 0]
    pi = rng.integers(0, len(pos), size=(n_boot, len(pos)))
    ni = rng.integers(0, len(neg), size=(n_boot, len(neg)))
    return auc_rows(pos[pi], neg[ni])


def observed(y, p, kind):
    if kind == "rmse":
        return float(np.sqrt(np.mean((p - y) ** 2)))
    return float(roc_auc_score(y, p))


def oriented(a, b, kind):
    """Positive = first stratum better."""
    return (b - a) if kind == "rmse" else (a - b)


df = pd.read_excel(XLSX, sheet_name="Per_Compound_Data", header=5)
df = df.dropna(subset=["endpoint"])

# --- sanity: rank identity must equal sklearn on the observed data ---------
for ep, (kind, tools) in SPEC.items():
    if kind != "auroc":
        continue
    sub = df[df["endpoint"] == ep]
    for tool, col in tools.items():
        s = sub[["ground_truth", col]].copy()
        s["ground_truth"] = num(s["ground_truth"]); s[col] = num(s[col])
        s = s.dropna()
        y, p = s["ground_truth"].values, s[col].values
        mine = float(auc_rows(p[y == 1][None, :], p[y == 0][None, :])[0])
        assert abs(mine - roc_auc_score(y, p)) < 1e-12, (ep, tool)
print("rank-identity AUROC matches sklearn on all observed data\n")

rows = []
master = np.random.default_rng(SEED)

for ep, (kind, tools) in SPEC.items():
    sub = df[df["endpoint"] == ep]
    for tool, col in tools.items():
        s = sub[["novelty_bin", "ground_truth", col]].copy()
        s["ground_truth"] = num(s["ground_truth"]); s[col] = num(s[col])
        s = s.dropna()

        point, draws, nn = {}, {}, {}
        for b in BINS:
            c = s[s["novelty_bin"] == b]
            y, p = c["ground_truth"].values, c[col].values
            nn[b] = len(y)
            point[b] = observed(y, p, kind)
            rng = np.random.default_rng(master.integers(2 ** 32))
            draws[b] = boot_draws(y, p, kind, rng)

        for a, bb in [("Near", "Novel"), ("Near", "Intermediate"),
                      ("Intermediate", "Novel")]:
            d = oriented(draws[a], draws[bb], kind)
            obs = oriented(point[a], point[bb], kind)
            lo, hi = np.percentile(d, [2.5, 97.5])
            pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
            rows.append(dict(
                endpoint=ep.split(" (")[0], tool=tool, metric=kind.upper(),
                comparison=f"{a} - {bb}", n_a=nn[a], n_b=nn[bb],
                val_a=round(point[a], 3), val_b=round(point[bb], 3),
                delta=round(float(obs), 3),
                ci_low=round(float(lo), 3), ci_high=round(float(hi), 3),
                p=round(float(max(min(pv, 1.0), 1.0 / N_BOOT)), 4)))

res = pd.DataFrame(rows)
res.to_csv("/home/claude/bootstrap_novelty_results.csv", index=False)

print("delta > 0 means BETTER on the first-named (more drug-like) stratum.")
print(f"{N_BOOT} stratified resamples, seed {SEED}.\n")
for comp in ["Near - Novel", "Near - Intermediate", "Intermediate - Novel"]:
    print("=" * 104)
    print(comp)
    print("=" * 104)
    for _, r in res[res["comparison"] == comp].iterrows():
        star = "  <-- p<0.05" if r["p"] < 0.05 else ""
        print(f" {r['endpoint']:11s} {r['tool']:13s} {r['metric']:5s} "
              f"n={r['n_a']:3d}/{r['n_b']:3d}  "
              f"{r['val_a']:6.3f} vs {r['val_b']:6.3f}  "
              f"d={r['delta']:+6.3f}  "
              f"CI[{r['ci_low']:+6.3f},{r['ci_high']:+6.3f}]  "
              f"p={r['p']:.4f}{star}")
    print()
