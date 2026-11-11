"""Paired fold-level comparison of each FAM ratio with the baseline.

Reads run_summary.csv (written by summarize_runs.py). For every model, metric, and FAM ratio,
the per-fold difference from the baseline trained on the same fold is computed, and the script
reports its mean (delta), 95% confidence interval (t-distribution), the number of folds in which
FAM scored higher (wins), and a two-sided paired t-test. p-values are Holm-Bonferroni adjusted
across the model-ratio comparisons of each metric (p_holm_within_metric), as in the paper, and
across all comparisons (p_holm_all).

Outputs (in --output_dir):
    paired_results.csv   one row per model x ratio x metric
    args.json            parameters of this run
Usage:
    python paired_analysis.py
"""
import argparse
import json
import os
from datetime import datetime

import pandas as pd

from utils import METRICS, holm, paired_stats, variant_to_ratio


def parse_args():
    p = argparse.ArgumentParser(description="Paired FAM vs. baseline analysis")
    p.add_argument("--summary", type=str, default="experiment_results/summary/run_summary.csv")
    p.add_argument("--baseline", type=str, default="dataset_kfold")
    p.add_argument("--variants", type=str, nargs="+",
                   default=["dataset_kfold_FDS_1_1", "dataset_kfold_FDS_1_3", "dataset_kfold_FDS_1_5"])
    p.add_argument("--models", type=str, nargs="+",
                   default=["yolov8s-seg", "yolo11s-seg", "yolo26s-seg"])
    p.add_argument("--metrics", type=str, nargs="+", default=METRICS)
    p.add_argument("--min_folds", type=int, default=3, help="Minimum number of paired folds for a comparison")
    p.add_argument("--output_dir", type=str, default="experiment_results/paired")
    p.add_argument("--log_dir", type=str, default="experiment_results/paired/logs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "run_time": datetime.now().isoformat()}, f, indent=2)

    df = pd.read_csv(args.summary)
    log, rows = [], []
    for model in args.models:
        b = df[(df.Dataset_Variant == args.baseline) & (df.Model == model)].set_index("Fold")
        if b.empty:
            log.append(f"SKIP {model}: no baseline runs")
            continue
        for var in args.variants:
            a = df[(df.Dataset_Variant == var) & (df.Model == model)].set_index("Fold")
            if a.empty:
                log.append(f"SKIP {model} {var}: no runs found")
                continue
            folds = sorted(set(b.index) & set(a.index))
            missing = sorted(set(b.index) ^ set(a.index))
            if missing:
                log.append(f"NOTE {model} {var}: folds without a pair are excluded: {', '.join(missing)}")
            if len(folds) < args.min_folds:
                log.append(f"SKIP {model} {var}: only {len(folds)} paired folds")
                continue
            for m in args.metrics:
                r = paired_stats(b.loc[folds, m].values, a.loc[folds, m].values)
                rows.append({"model": model, "variant": var, "ratio": variant_to_ratio(var), "metric": m, **r})

    if not rows:
        log.append("No comparisons could be made; check --summary, --baseline, --variants, and --models.")
        print("\n".join(log))
        return

    res = pd.DataFrame(rows)
    res["p_holm_within_metric"] = res.groupby("metric")["p_value"].transform(lambda s: holm(s.values))
    res["p_holm_all"] = holm(res["p_value"].values)
    res.to_csv(os.path.join(args.output_dir, "paired_results.csv"), index=False)

    n_cmp = res.groupby("metric").size().iloc[0]
    log.append(f"{len(res)} comparisons ({n_cmp} model-ratio comparisons per metric) -> {args.output_dir}")
    if res.p_holm_within_metric.notna().any():
        log.append(f"Smallest Holm-adjusted p-value (within metric): {res.p_holm_within_metric.min():.3f}")
    n_nan = int(res.p_value.isna().sum())
    if n_nan:
        log.append(f"NOTE {n_nan} comparison(s) without a p-value (identical paired differences)")
    with open(os.path.join(args.log_dir, "paired_analysis_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
