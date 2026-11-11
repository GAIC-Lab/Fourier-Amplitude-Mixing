import os
import csv
import argparse
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

def initialize_csv_logger(csv_path):
    """
    Creates and initializes the CSV file header for tracking training and validation metrics.
    Does not overwrite if the file already exists (useful for resuming folds).
    """
    headers = [
        'Dataset_Variant', 'Model_Architecture', 'Fold', 'Epoch',
        'Train_Box_Loss', 'Train_Seg_Loss', 'Train_Class_Loss',
        'Val_Box_Loss', 'Val_Seg_Loss', 'Val_Class_Loss',
        'Box_Precision', 'Box_Recall', 'Box_mAP50', 'Box_mAP50_95',
        'Mask_Precision', 'Mask_Recall', 'Mask_mAP50', 'Mask_mAP50_95'
    ]
    
    file_exists = os.path.isfile(csv_path)
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
    if not file_exists:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_metrics_to_csv(csv_path, row_data):
    """
    Appends a single epoch metric record to the CSV file.
    """
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

def extract_and_log_results(results_csv_path, log_csv_path, dataset_variant, model_arch, fold_num):
    """
    Parses Ultralytics internal results.csv and logs standard metrics to our master CSV.
    """
    if not os.path.exists(results_csv_path):
        print(f"\nWarning: Results file not found at {results_csv_path}")
        return

    with open(results_csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items()}
            
            # Epoch & Training Losses
            epoch = clean_row.get('epoch', '0')
            t_box = clean_row.get('train/box_loss', '0.0')
            t_seg = clean_row.get('train/seg_loss', '0.0')
            t_cls = clean_row.get('train/cls_loss', '0.0')
            
            # Validation Losses
            v_box = clean_row.get('val/box_loss', '0.0')
            v_seg = clean_row.get('val/seg_loss', '0.0')
            v_cls = clean_row.get('val/cls_loss', '0.0')
            
            # Bounding Box Metrics
            b_p = clean_row.get('metrics/precision(B)', '0.0')
            b_r = clean_row.get('metrics/recall(B)', '0.0')
            b_m50 = clean_row.get('metrics/mAP50(B)', '0.0')
            b_m95 = clean_row.get('metrics/mAP50-95(B)', '0.0')

            # Mask Segmentation Metrics
            m_p = clean_row.get('metrics/precision(M)', '0.0')
            m_r = clean_row.get('metrics/recall(M)', '0.0')
            m_m50 = clean_row.get('metrics/mAP50(M)', '0.0')
            m_m95 = clean_row.get('metrics/mAP50-95(M)', '0.0')

            row_data = [
                dataset_variant, model_arch, f"fold_{fold_num}", epoch,
                t_box, t_seg, t_cls, v_box, v_seg, v_cls,
                b_p, b_r, b_m50, b_m95, m_p, m_r, m_m50, m_m95
            ]
            append_metrics_to_csv(log_csv_path, row_data)

def train_single_fold(model_type, dataset_yaml, project_name, run_name, epochs=100, batch_size=8, img_size=640):
    """
    Initializes and trains a YOLO segmentation model for a single fold.
    """
    model = YOLO(model_type)
    
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project_name,
        name=run_name,
        exist_ok=True,
        verbose=False,
        save=True
    )
    # Folder where Ultralytics actually saved this run (contains results.csv and weights/)
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    return results, save_dir

def run_kfold_experiment(dataset_dir, model_type="yolo11s-seg.pt", epochs=100, batch_size=8, num_folds=5, start_fold=1):
    """
    Executes the K-fold training loop, allowing for interruption and resumption.
    """
    dataset_variant = os.path.basename(os.path.normpath(dataset_dir))
    model_arch = os.path.splitext(os.path.basename(model_type))[0] 
    
    master_csv = os.path.join("experiment_results", f"{dataset_variant}_{model_arch}_metrics.csv")
    initialize_csv_logger(master_csv)
    
    print(f"\n=========================================")
    print(f"Starting K-Fold Training: {dataset_variant}")
    print(f"Model: {model_type} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Starting at Fold: {start_fold}")
    print(f"=========================================\n")
    
    fold_pbar = tqdm(range(start_fold, num_folds + 1), desc="K-Fold Progress", unit="fold")
    
    for fold in fold_pbar:
        fold_pbar.set_postfix({"Current Fold": f"{fold}/{num_folds}"})
        
        fold_dir = os.path.join(dataset_dir, f"fold_{fold}")
        yaml_path = os.path.join(fold_dir, "dataset.yaml")
        
        # Fixed pathing: matches Ultralytics default segment structure
        project_dir = os.path.join("runs", "segment", "runs_kfold", dataset_variant)
        run_name = f"fold_{fold}_{model_arch}"
        
        _, save_dir = train_single_fold(
            model_type=model_type,
            dataset_yaml=yaml_path,
            project_name=project_dir,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Use the run folder reported by Ultralytics; fall back to the expected path
        run_dir = str(save_dir) if save_dir else os.path.join(project_dir, run_name)
        yolo_results_csv = os.path.join(run_dir, "results.csv")
        extract_and_log_results(yolo_results_csv, master_csv, dataset_variant, model_arch, fold)

    print(f"\n✅ Training complete for {dataset_variant}. Master metrics saved to: {master_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold YOLO Segmentation Training")
    parser.add_argument("--target_dataset", type=str, required=True, help="Path to the dataset folder (e.g., dataset_kfold)")
    parser.add_argument("--model_architecture", type=str, required=True, help="Model architecture file (e.g., yolo11s-seg.pt)")
    parser.add_argument("--start_fold", type=int, default=1, help="Fold to start training from (1-5)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    
    args = parser.parse_args()
    
    run_kfold_experiment(
        dataset_dir=args.target_dataset,
        model_type=args.model_architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        start_fold=args.start_fold
    )
    