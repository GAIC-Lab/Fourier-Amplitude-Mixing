"""Summarize the validation metrics of every training run into one CSV.

Reads the Ultralytics results.csv of each run written by train_yolo_kfold.py:
    <runs_dir>/<variant>/fold_<k>_<model>/results.csv
and records, for each run, the metrics at the epoch saved as best.pt (highest Ultralytics fitness).

Outputs (in --output_dir):
    run_summary.csv       one row per variant x model x fold
    run_summary_mean_sd.csv   mean and standard deviation over folds per variant x model
    args.json             parameters of this run
Usage:
    python summarize_runs.py
    python summarize_runs.py --runs_dir runs/segment/runs_kfold --output_dir experiment_results/summary
"""
import argparse
import json
import os
from datetime import datetime
from glob import glob

import pandas as pd

from utils import METRICS, parse_run_path, read_results, summarize_run, variant_to_ratio


def parse_args():
    p = argparse.ArgumentParser(description="Summarize best-epoch validation metrics of all runs")
    p.add_argument("--runs_dir", type=str, default="runs/segment/runs_kfold",
                   help="Folder searched recursively for results.csv files")
    p.add_argument("--w_map50", type=float, default=0.1, help="Fitness weight of mAP@0.5 (Ultralytics default)")
    p.add_argument("--w_map50_95", type=float, default=0.9, help="Fitness weight of mAP@0.5:0.95 (Ultralytics default)")
    p.add_argument("--output_dir", type=str, default="experiment_results/summary")
    p.add_argument("--log_dir", type=str, default="experiment_results/summary/logs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "run_time": datetime.now().isoformat()}, f, indent=2)

    log, rows = [], []
    for path in sorted(glob(os.path.join(args.runs_dir, "**", "results.csv"), recursive=True)):
        meta = parse_run_path(path)
        if meta is None:
            log.append(f"SKIP (folder name is not fold_<k>_<model>): {path}")
            continue
        try:
            row = summarize_run(read_results(path), args.w_map50, args.w_map50_95)
        except (KeyError, ValueError) as e:
            log.append(f"SKIP (unexpected columns or empty file: {e}): {path}")
            continue
        variant, fold, model = meta
        rows.append({"Dataset_Variant": variant, "Ratio": variant_to_ratio(variant), "Model": model,
                     "Fold": f"fold_{fold}", "Source": path, **row})

    if not rows:
        log.append(f"No results.csv found under {args.runs_dir}")
        print("\n".join(log))
        return

    df = pd.DataFrame(rows)
    front = ["Dataset_Variant", "Ratio", "Model", "Fold", "epochs_trained", "best_epoch", "best_fitness"] + METRICS
    df = df[front + [c for c in df.columns if c not in front]]
    df = df.sort_values(["Model", "Dataset_Variant", "Fold"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.output_dir, "run_summary.csv"), index=False)

    keys = ["Model", "Dataset_Variant", "Ratio"]
    agg = df.groupby(keys)[METRICS + ["best_epoch", "sec_per_epoch"]].agg(["mean", "std"]).round(4)
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_folds", df.groupby(keys).size())
    agg.reset_index().to_csv(os.path.join(args.output_dir, "run_summary_mean_sd.csv"), index=False)

    # completeness checks
    folds = sorted(df.Fold.unique())
    for (model, variant), g in df.groupby(["Model", "Dataset_Variant"]):
        missing = sorted(set(folds) - set(g.Fold))
        if missing:
            log.append(f"MISSING {model} {variant}: {', '.join(missing)}")
    for _, r in df[df.duplicated(["Dataset_Variant", "Model", "Fold"], keep=False)].iterrows():
        log.append(f"DUPLICATE {r.Model} {r.Dataset_Variant} {r.Fold}: {r.Source}")
    for _, r in df[df.epochs_trained < df.epochs_trained.max()].iterrows():
        log.append(f"WARNING short run ({r.epochs_trained} epochs): {r.Source}")
    for _, r in df[df.n_nan_values > 0].iterrows():
        log.append(f"NOTE {r.n_nan_values} NaN value(s): {r.Source}")

    log.append(f"Summarized {len(df)} runs -> {args.output_dir}")
    with open(os.path.join(args.log_dir, "summarize_runs_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
