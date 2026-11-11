# Model-Dependent Effects of Fourier Amplitude Mixing on Roadside Parking Space Segmentation

Code for the paper *Model-Dependent Effects of Fourier Amplitude Mixing on Roadside Parking Space Segmentation* (under review).

The study evaluates **Fourier amplitude mixing (FAM)**, an intra-dataset variant of Fourier domain adaptation that blends the low-frequency amplitude spectra of two training images while keeping the phase of one, as data augmentation for instance segmentation of roadside parking spaces in dashcam images. Three YOLO generations (YOLOv8s-seg, YOLO11s-seg, YOLO26s-seg) are trained with FAM expansion ratios of 1:1, 1:3, and 1:5 and compared with unaugmented baselines under stratified 5-fold cross-validation.

> **Naming note.** Folder and file names in this repository use **FDS** (for example `dataset_kfold_FDS_1_3`). This is the method called **FAM** in the paper: `FDS_1_3` = FAM 1:3.

---

## Contents

| File | Purpose |
|---|---|
| `utils_dataset_stats.py` | Class statistics of the full dataset (instances, images per class, mean area) |
| `utils_kfold_split.py` | Creates the five folds in YOLO format, recreating the paper's folds from `folds.csv` |
| `utils_check_kfold_stats.py` | Class statistics of every fold and split → `kfold_dataset_stats.csv` |
| `utils_generate_fds.py` | Generates the FAM training sets (`*_FDS_1_1`, `*_FDS_1_3`, `*_FDS_1_5`) for each fold |
| `train_yolo_kfold.py` | Trains a YOLO segmentation model on all five folds of one dataset variant |
| `summarize_runs.py` | Reads every run's training log and summarizes the best-epoch metrics into one CSV |
| `paired_analysis.py` | Paired fold-level comparison of each FAM ratio with the baseline (Δ, 95% CI, wins, paired *t*-test, Holm correction) |
| `utils.py` | Helper functions for `summarize_runs.py` and `paired_analysis.py` |

---

## Installation

```bash
git clone https://github.com/GAIC-Lab/Fourier-Amplitude-Mixing.git
cd Fourier-Amplitude-Mixing
pip install -r requirements.txt
```

The experiments in the paper used Ultralytics 8.4.70 with an NVIDIA GeForce RTX 3090 GPU. Install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org).

---

## Data

The dataset is available on Zenodo: **[Zenodo DOI link]**

It contains the 544 dashcam images and polygon annotations originally published in Ehsan and Chen (2025) on [Roboflow Universe](https://universe.roboflow.com/ehsaanullah/free-parking-space-detection-e7hqm/dataset/9) under the CC BY 4.0 license, together with the day/night labels and the fold assignments used in the paper. Download it and place it in `dataset/`:

```
dataset/
├── images/        # 544 images (*.jpg)
├── labels/        # YOLO segmentation labels (*.txt), same base names as the images
├── day.txt        # names of the 353 daytime images, one per line
├── night.txt      # names of the 191 night-time images, one per line
└── folds.csv      # validation fold (1-5) of every image, as used in the paper
```

The labels use four classes. The paper uses short names; the original Roboflow export uses longer names for the same class IDs:

| ID | Name in the paper | Roboflow name | Description |
|---|---|---|---|
| 0 | `slot_empty` | `car-parking-available` | Unoccupied designated car space enclosed by a white painted box |
| 1 | `slot_occ` | `car-parking-occupied` | Occupied designated car space |
| 2 | `space_free` | `free-space-parking` | Kerb with a solid white edge line but no box, where parking is generally permitted |
| 3 | `no_park` | `restricted-area` | No-parking zone marked by red or yellow kerb lines |

The scripts below create the training folders:

```
dataset/
├── dataset_kfold/fold_{1..5}/{train,val}/{images,labels}, dataset.yaml
├── dataset_kfold_FDS_1_1/fold_{1..5}/...     # FAM 1:1 (validation sets unchanged)
├── dataset_kfold_FDS_1_3/fold_{1..5}/...     # FAM 1:3
└── dataset_kfold_FDS_1_5/fold_{1..5}/...     # FAM 1:5
```

---

## Usage

### 1. Inspect the dataset

```bash
python utils_dataset_stats.py
```

Prints the number of instances, the number of images containing each class, and the average area of each class.

### 2. Create the folds

```bash
python utils_kfold_split.py
python utils_check_kfold_stats.py      # optional: per-fold class statistics -> kfold_dataset_stats.csv
```

This copies the images and labels into the five folds in YOLO format and writes each fold's `dataset.yaml`. With `dataset/folds.csv` present, it recreates **exactly the folds used in the paper**. Without it, the images are split with stratified 5-fold cross-validation on the presence of `slot_empty`, the rarest class (`random_state=42`, set with `--seed`), and the resulting assignment is saved as `folds.csv`.

The script checks that every image has a label file and appears in exactly one of `day.txt` and `night.txt`. It writes `fold_summary.csv` with the number of images, `slot_empty` images, and daytime and night-time images in each fold and split. Run `python utils_kfold_split.py --help` for all options.

### 3. Generate the FAM training sets

```bash
python utils_generate_fds.py
```

For every fold and ratio (1:1, 1:3, 1:5), this copies the original training and validation images, adds 1, 3, or 5 augmented images per training image, and writes a `dataset.yaml` pointing to the new folders. The defaults are the settings used in the paper:

| Option | Default | Meaning |
|---|---|---|
| `--ratios` | `1 3 5` | Augmented images per original training image |
| `--alpha_min`, `--alpha_max` | `0.2`, `0.8` | Range of the blending coefficient α, drawn per augmented image |
| `--beta` | `0.01` | Size of the low-frequency region: a square of (2b+1) × (2b+1) frequencies with b = ⌊β · min(H, W)⌋ |
| `--seed` | `-1` | Random seed; `-1` leaves the augmentation unseeded |

Style images are drawn uniformly at random, with replacement, from the **same fold's training set**, so validation images are never used. The annotations of each original image are reused for its augmented copies, because FAM keeps the image phase and therefore the positions and shapes of the parking regions. The parameters of each run are saved as `args.json` in the output folder.

The augmentation in the paper was not seeded, so regenerated images will differ slightly from those used in the paper. Use `--seed` (for example `--seed 0`) to make future runs reproducible.

### 4. Train

Train one model on all five folds of one dataset variant:

```bash
python train_yolo_kfold.py --target_dataset dataset/dataset_kfold --model_architecture yolo11s-seg.pt
```

The paper's 60 runs (3 models × 4 training conditions × 5 folds) can be run with:

```bash
for data in dataset_kfold dataset_kfold_FDS_1_1 dataset_kfold_FDS_1_3 dataset_kfold_FDS_1_5; do
  for model in yolov8s-seg.pt yolo11s-seg.pt yolo26s-seg.pt; do
    python train_yolo_kfold.py --target_dataset dataset/$data --model_architecture $model
  done
done
```

Training uses the default Ultralytics settings, including COCO-pretrained weights, the default augmentation pipeline, and `optimizer=auto`, with 100 epochs, batch size 8, and an input size of 640 × 640 (the script's defaults). Use `--start_fold` to resume an interrupted run.

**Outputs**

- `runs/segment/runs_kfold/<variant>/fold_<k>_<model>/`: the Ultralytics run folder, including `results.csv` (validation metrics per epoch) and `weights/best.pt`, the checkpoint with the highest validation fitness;
- `experiment_results/<variant>_<model>_metrics.csv`: the per-epoch losses and metrics of all five folds collected in one file.

### 5. Summarize the runs

```bash
python summarize_runs.py
```

This reads the `results.csv` of every run under `runs/segment/runs_kfold/` and records, for each run, the validation metrics at the epoch saved as `best.pt`: the first epoch with the highest Ultralytics fitness, 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95, summed over box and mask. It writes to `experiment_results/summary/`:

| File | Content |
|---|---|
| `run_summary.csv` | One row per dataset variant × model × fold: selected epoch, its fitness, box and mask precision, recall, mAP@0.5 and mAP@0.5:0.95, seconds per epoch, and the number of missing values in the log |
| `run_summary_mean_sd.csv` | Mean and standard deviation over the folds for each variant × model, with the number of folds |
| `args.json` | Parameters of the run |

The `Ratio` column gives `baseline`, `1:1`, `1:3`, or `1:5`. The log in `experiment_results/summary/logs/` reports missing folds, duplicate runs, runs shorter than the others, and missing values. `run_summary.csv` is the input for the paired analysis below and for any further analysis.

### 6. Paired analysis

```bash
python paired_analysis.py
```

Because every training condition uses the same five folds, each FAM model is compared with the baseline model trained and validated on the same fold. For every model, FAM ratio, and metric, the script computes the per-fold difference (FAM minus baseline) and writes `experiment_results/paired/paired_results.csv` with:

| Column | Meaning |
|---|---|
| `baseline_mean`, `fam_mean` | Mean over the folds without and with FAM |
| `mean_diff` | Mean paired difference Δ |
| `ci95_low`, `ci95_high` | 95% confidence interval of Δ (*t*-distribution) |
| `wins` | Number of folds in which FAM scored higher (out of five) |
| `per_fold_diff` | The individual per-fold differences |
| `t`, `p_value` | Two-sided paired *t*-test |
| `p_holm_within_metric` | *p*-value adjusted with the Holm–Bonferroni procedure across the 9 model–ratio comparisons of each metric (as in the paper) |
| `p_holm_all` | *p*-value adjusted across all comparisons |
| `cohens_dz` | Paired effect size (Δ divided by the standard deviation of the differences) |

These are the values reported in Table 3 of the paper, which shows the three mask metrics (mask mAP@0.5:0.95, mask mAP@0.5, and mask recall). The file covers all eight metrics, so its smallest Holm-adjusted *p*-value (0.096, for a box metric) is lower than the smallest among the mask metrics reported in the paper (0.106). With only five folds the tests have limited power, which is why the consistency of the direction of change (`wins`) is reported alongside the *p*-values.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{subrpa2026fam,
  title   = {Model-Dependent Effects of Fourier Amplitude Mixing on Roadside Parking Space Segmentation},
  author  = {Sub-r-pa, Chayanon and Ehsan, Muhammad and Chen, Rung-Ching},
  journal = {Under review},
  year    = {2026}
}
```

Please also cite the original dataset paper:

> Ehsan, M., Chen, R.-C. (2025). Smart Detection of Open Parking Spaces in Urban Taiwan Using YOLOv8 Segmentation on Dashcam Footage. In: *International Conference on Intelligent Information Hiding and Multimedia Signal Processing*, pp. 356–367. Springer. https://doi.org/10.1007/978-3-032-22592-4_33

---

## License

The dataset is distributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license of the original Roboflow dataset; please cite Ehsan and Chen (2025) when using it.

## Acknowledgements

This work was supported by the National Science and Technology Council, Taiwan, R.O.C., under grants NSTC-115-2221-E-324-012- and NSTC-114-2637-E-324-003-.
