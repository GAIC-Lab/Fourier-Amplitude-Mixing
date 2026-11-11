"""Helper functions for summarize_runs.py."""
import os
import re

import pandas as pd

# Ultralytics results.csv column -> name used in the summary
METRIC_COLUMNS = {
    "Box_Precision": "metrics/precision(B)",
    "Box_Recall": "metrics/recall(B)",
    "Box_mAP50": "metrics/mAP50(B)",
    "Box_mAP50_95": "metrics/mAP50-95(B)",
    "Mask_Precision": "metrics/precision(M)",
    "Mask_Recall": "metrics/recall(M)",
    "Mask_mAP50": "metrics/mAP50(M)",
    "Mask_mAP50_95": "metrics/mAP50-95(M)",
}
METRICS = list(METRIC_COLUMNS)

RUN_DIR = re.compile(r"^fold_(\d+)_(.+)$")   # run folder written by train_yolo_kfold.py, e.g. fold_1_yolo11s-seg


def parse_run_path(results_csv):
    """runs/.../<variant>/fold_<k>_<model>/results.csv -> (variant, fold, model), or None."""
    run_dir = os.path.basename(os.path.dirname(results_csv))
    m = RUN_DIR.match(run_dir)
    if not m:
        return None
    variant = os.path.basename(os.path.dirname(os.path.dirname(results_csv)))
    return variant, int(m.group(1)), m.group(2)


def variant_to_ratio(variant):
    """dataset_kfold -> 'baseline'; dataset_kfold_FDS_1_3 -> '1:3' (FDS = FAM in the paper)."""
    m = re.search(r"FDS_(\d+)_(\d+)$", variant)
    return f"{m.group(1)}:{m.group(2)}" if m else "baseline"


def read_results(path):
    """Read an Ultralytics results.csv; strips the padding Ultralytics adds to column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def fitness(df, w_map50=0.1, w_map50_95=0.9):
    """Ultralytics segmentation fitness used to choose best.pt: box fitness + mask fitness."""
    return (w_map50 * df[METRIC_COLUMNS["Box_mAP50"]] + w_map50_95 * df[METRIC_COLUMNS["Box_mAP50_95"]]
            + w_map50 * df[METRIC_COLUMNS["Mask_mAP50"]] + w_map50_95 * df[METRIC_COLUMNS["Mask_mAP50_95"]])


def summarize_run(df, w_map50=0.1, w_map50_95=0.9):
    """Metrics at the epoch saved as best.pt (first epoch with the highest fitness)."""
    fit = fitness(df, w_map50, w_map50_95)
    k = fit.fillna(-1).idxmax()          # best.pt is replaced only on strict improvement -> first maximum
    row = {
        "epochs_trained": int(df["epoch"].max()),
        "best_epoch": int(df.loc[k, "epoch"]),
        "best_fitness": round(float(fit.loc[k]), 4),
    }
    for name, col in METRIC_COLUMNS.items():
        row[name] = round(float(df.loc[k, col]), 4)
    row["sec_per_epoch"] = round(float(df["time"].diff().median()), 2) if "time" in df else float("nan")
    row["n_nan_values"] = int(df.select_dtypes("number").isna().sum().sum())
    return row


# ---------------------------------------------------------------------------
# Paired statistics (paired_analysis.py)
# ---------------------------------------------------------------------------
import numpy as np
from scipy import stats


def holm(p):
    """Holm-Bonferroni adjusted p-values. Missing p-values (NaN, e.g. when all paired
    differences are identical) are left missing and do not count as comparisons."""
    p = np.asarray(p, dtype=float)
    adj = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    m, running = len(valid), 0.0
    for i, k in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (m - i) * p[k])
        adj[k] = min(running, 1.0)
    return adj


def paired_stats(base, aug):
    """Paired comparison of per-fold scores (augmented minus baseline on the same folds)."""
    diff = aug - base
    n = len(diff)
    mean, sd = diff.mean(), diff.std(ddof=1)
    t, p = stats.ttest_rel(aug, base)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {
        "n_folds": n,
        "baseline_mean": base.mean(), "fam_mean": aug.mean(),
        "mean_diff": mean, "sd_diff": sd,
        "ci95_low": mean - half, "ci95_high": mean + half,
        "t": t, "p_value": p,
        "cohens_dz": mean / sd if sd > 0 else np.nan,
        "wins": int((diff > 0).sum()),
        "per_fold_diff": ";".join(f"{v:+.4f}" for v in diff),
    }
