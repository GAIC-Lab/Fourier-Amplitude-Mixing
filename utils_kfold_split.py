"""Create the 5-fold cross-validation datasets in YOLO format.

Input (as released on Zenodo):
    <data_root>/images/*.jpg     the 544 images
    <data_root>/labels/*.txt     YOLO segmentation labels (same base names)
    <data_root>/day.txt          names of the daytime images, one per line
    <data_root>/night.txt        names of the night-time images, one per line
    <data_root>/folds.csv        validation fold of every image (columns: image, val_fold)

If folds.csv exists, the folds of the paper are recreated exactly. Otherwise the images are split
with StratifiedKFold (stratified on the presence of slot_empty, shuffle=True, random_state=--seed)
and the resulting assignment is written to <output_dir>/folds.csv.

Output:
    <output_dir>/fold_<k>/train/{images,labels}, val/{images,labels}, dataset.yaml
    <output_dir>/fold_summary.csv   images, slot_empty images, and day/night images per fold and split
    <output_dir>/args.json
Usage:
    python utils_kfold_split.py
    python utils_kfold_split.py --data_root dataset --output_dir dataset/dataset_kfold
"""
import argparse
import json
import os
import shutil
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Class names used in the paper. The Roboflow export uses the same class IDs with other names:
# 0 car-parking-available, 1 car-parking-occupied, 2 free-space-parking, 3 restricted-area.
CLASS_NAMES = ["slot_empty", "slot_occ", "space_free", "no_park"]


def parse_args():
    p = argparse.ArgumentParser(description="Stratified 5-fold split in YOLO format")
    p.add_argument("--data_root", type=str, default="dataset",
                   help="Folder with images/, labels/, day.txt, night.txt, and folds.csv")
    p.add_argument("--folds_file", type=str, default="",
                   help="Fold assignment CSV (default: <data_root>/folds.csv if it exists)")
    p.add_argument("--output_dir", type=str, default="dataset/dataset_kfold")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42, help="random_state of StratifiedKFold (used without folds.csv)")
    p.add_argument("--minority_class_id", type=int, default=0, help="Class used for stratification (slot_empty)")
    p.add_argument("--log_dir", type=str, default="dataset/dataset_kfold/logs")
    return p.parse_args()


def read_list(path):
    """Read a day.txt / night.txt file: one image name per line."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def has_class(label_path, class_id):
    if not os.path.exists(label_path):
        return 0
    with open(label_path) as f:
        return int(any(line.split() and int(line.split()[0]) == class_id for line in f))


def label_path_for(lbl_dir, image_name):
    return os.path.join(lbl_dir, os.path.splitext(image_name)[0] + ".txt")


def assign_folds(names, labels_dir, folds_file, args, log):
    """Validation fold (1..n_splits) of every image."""
    if os.path.exists(folds_file):
        f = pd.read_csv(folds_file)
        mapping = dict(zip(f["image"], f["val_fold"]))
        missing = [n for n in names if n not in mapping]
        extra = sorted(set(mapping) - set(names))
        if missing:
            raise ValueError(f"{len(missing)} images are not in {folds_file}, e.g. {missing[:3]}")
        if extra:
            log.append(f"NOTE {len(extra)} names in {folds_file} have no image and are ignored")
        log.append(f"Folds taken from {folds_file} (the folds used in the paper)")
        return np.array([int(mapping[n]) for n in names])

    # Same procedure as the original split: sorted image list, stratified on the minority class
    y = np.array([has_class(label_path_for(labels_dir, n), args.minority_class_id) for n in names])
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    val_fold = np.zeros(len(names), dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(names, y), start=1):
        val_fold[val_idx] = k
    log.append(f"No folds file found; folds created with StratifiedKFold(random_state={args.seed})")
    return val_fold


def write_yaml(fold_dir):
    with open(os.path.join(fold_dir, "dataset.yaml"), "w") as f:
        f.write(f"train: {os.path.abspath(os.path.join(fold_dir, 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(fold_dir, 'val'))}\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "run_time": datetime.now().isoformat()}, f, indent=2)

    img_dir = os.path.join(args.data_root, "images")
    lbl_dir = os.path.join(args.data_root, "labels")
    names = sorted(os.path.basename(p) for p in glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))
    if not names:
        raise FileNotFoundError(f"No images found in {img_dir}")
    log = [f"{len(names)} images in {img_dir}"]

    no_label = [n for n in names if not os.path.exists(label_path_for(lbl_dir, n))]
    if no_label:
        log.append(f"WARNING {len(no_label)} images have no label file, e.g. {no_label[:3]}")
    day = read_list(os.path.join(args.data_root, "day.txt"))
    night = read_list(os.path.join(args.data_root, "night.txt"))
    if day or night:
        unlisted = [n for n in names if n not in day and n not in night]
        both = sorted(day & night)
        log.append(f"day.txt: {len(day)} images, night.txt: {len(night)} images")
        if unlisted:
            log.append(f"WARNING {len(unlisted)} images are in neither day.txt nor night.txt, e.g. {unlisted[:3]}")
        if both:
            log.append(f"WARNING {len(both)} images are in both day.txt and night.txt, e.g. {both[:3]}")

    folds_file = args.folds_file or os.path.join(args.data_root, "folds.csv")
    val_fold = assign_folds(names, lbl_dir, folds_file, args, log)
    if not os.path.exists(folds_file):
        pd.DataFrame({"image": names, "val_fold": val_fold}).to_csv(
            os.path.join(args.output_dir, "folds.csv"), index=False)

    summary = []
    for k in range(1, args.n_splits + 1):
        fold_dir = os.path.join(args.output_dir, f"fold_{k}")
        for split in ("train", "val"):
            sel = [n for n, v in zip(names, val_fold) if (v == k) == (split == "val")]
            os.makedirs(os.path.join(fold_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, split, "labels"), exist_ok=True)
            for n in tqdm(sel, desc=f"fold {k} {split}", leave=False):
                shutil.copy(os.path.join(img_dir, n), os.path.join(fold_dir, split, "images", n))
                lp = label_path_for(lbl_dir, n)
                if os.path.exists(lp):
                    shutil.copy(lp, os.path.join(fold_dir, split, "labels", os.path.basename(lp)))
            summary.append({"fold": k, "split": split, "images": len(sel),
                            "slot_empty_images": sum(has_class(label_path_for(lbl_dir, n), args.minority_class_id) for n in sel),
                            "day_images": sum(n in day for n in sel), "night_images": sum(n in night for n in sel)})
        write_yaml(fold_dir)

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(args.output_dir, "fold_summary.csv"), index=False)
    log.append("Per-fold summary:\n" + s.to_string(index=False))
    log.append(f"Folds written to {args.output_dir}")
    with open(os.path.join(args.log_dir, "kfold_split_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
