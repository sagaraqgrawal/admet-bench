"""
evaluate.py — metrics, bootstrap confidence intervals and pairwise significance.

Implements every statistic reported in the manuscript:

  Regression (solubility, secondary BBB logBB)
      RMSE, MAE, Spearman rho, mean signed error (bias)

  Classification (hERG, BBB, DILI)
      AUROC, AUPRC (probabilistic outputs only)
      MCC, sensitivity, specificity (all tools, including binary-only)

  Uncertainty
      95% CIs by non-parametric bootstrap (1,000 resamples, fixed seed)

  Pairwise comparison
      DeLong test for two correlated ROC curves

All metrics can be computed overall or stratified by novelty bin.

Usage
-----
    python evaluate.py --predictions ../results/per_compound_predictions.csv \
        --endpoint "hERG (blocker@10uM)" --tool-cols ADMETlab3.0 ADMET-AI

Requirements: pandas, numpy, scikit-learn, scipy
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)

N_BOOTSTRAP = 1000
SEED = 42
PROB_THRESHOLD = 0.5      # probability -> binary label
LOGBB_CUTOFF = -1.0       # pkCSM logBB -> BBB penetrant label


# ----------------------------------------------------------------- regression
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def regression_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "n": len(y),
        "RMSE": rmse(y, p),
        "MAE": float(np.mean(np.abs(y - p))),
        "Spearman": float(spearmanr(y, p).correlation),
        "bias": float(np.mean(p - y)),
    }


# ------------------------------------------------------------- classification
def classification_metrics(y, score=None, label=None):
    y = np.asarray(y, int)
    out = {"n": len(y), "n_pos": int(y.sum())}
    if score is not None and len(np.unique(y)) == 2:
        out["AUROC"] = float(roc_auc_score(y, score))
        out["AUPRC"] = float(average_precision_score(y, score))
    if label is not None:
        label = np.asarray(label, int)
        out["MCC"] = float(matthews_corrcoef(y, label))
        tn, fp, fn, tp = confusion_matrix(y, label, labels=[0, 1]).ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return out


# ------------------------------------------------------------------ bootstrap
def bootstrap_ci(fn, *arrays, n=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI for any metric function fn(*arrays)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    m = len(arrays[0])
    stats = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            v = fn(*[a[idx] for a in arrays])
            if v is not None and not np.isnan(v):
                stats.append(v)
        except Exception:
            continue
    if not stats:
        return (np.nan, np.nan)
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------- DeLong test
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_test(y, score1, score2):
    """
    DeLong test for two correlated ROC curves.
    Returns (auc1, auc2, z, p). Two-sided p-value.
    """
    y = np.asarray(y, int)
    s1 = np.asarray(score1, float)
    s2 = np.asarray(score2, float)

    pos1, neg1 = s1[y == 1], s1[y == 0]
    pos2, neg2 = s2[y == 1], s2[y == 0]
    m, n = len(pos1), len(neg1)
    preds = [np.r_[pos1, neg1], np.r_[pos2, neg2]]
    aucs = np.array([roc_auc_score(y, s) for s in (s1, s2)])

    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n

    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], np.nan, np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ------------------------------------------------------------------- driver
def score_endpoint(df, endpoint, tool_cols, kind="classification",
                   stratify=True):
    """Compute metrics for one endpoint, overall and per novelty bin."""
    sub = df[df["endpoint"] == endpoint].copy()
    rows = []
    groups = [("ALL", sub)]
    if stratify and "novelty_bin" in sub.columns:
        for b in ["Near", "Intermediate", "Novel"]:
            groups.append((b, sub[sub["novelty_bin"] == b]))

    for gname, g in groups:
        for tool in tool_cols:
            d = g[["ground_truth", tool]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(d) < 3:
                continue
            y = d["ground_truth"].values
            v = d[tool].values
            if kind == "regression":
                m = regression_metrics(y, v)
                lo, hi = bootstrap_ci(rmse, y, v)
                m["RMSE_CI"] = (round(lo, 3), round(hi, 3))
            else:
                label = (v >= PROB_THRESHOLD).astype(int)
                m = classification_metrics(y, score=v, label=label)
                if "AUROC" in m:
                    lo, hi = bootstrap_ci(
                        lambda yy, ss: roc_auc_score(yy, ss)
                        if len(np.unique(yy)) == 2 else np.nan, y, v)
                    m["AUROC_CI"] = (round(lo, 3), round(hi, 3))
            m.update(endpoint=endpoint, tool=tool, bin=gname)
            rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="per-compound predictions CSV")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--tool-cols", nargs="+", required=True)
    ap.add_argument("--kind", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--delong", nargs=2, default=None,
                    metavar=("TOOL_A", "TOOL_B"),
                    help="run DeLong test between two probabilistic tools")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    res = score_endpoint(df, args.endpoint, args.tool_cols, kind=args.kind)
    print(res.to_string(index=False))

    if args.delong:
        a, b = args.delong
        sub = df[df["endpoint"] == args.endpoint][["ground_truth", a, b]]
        sub = sub.apply(pd.to_numeric, errors="coerce").dropna()
        auc1, auc2, z, p = delong_test(sub["ground_truth"].values,
                                       sub[a].values, sub[b].values)
        print(f"\nDeLong: {a} AUROC={auc1:.3f} vs {b} AUROC={auc2:.3f} "
              f"| z={z:.2f} p={p:.4f}")


if __name__ == "__main__":
    main()
