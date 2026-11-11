import os
import shutil
from glob import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def get_minority_class_distribution(label_files, minority_class_id=0):
    """
    Scans label files to determine which images contain the minority class.
    Returns a binary array: 1 if the minority class is present, 0 otherwise.
    """
    y_stratify = []
    for label_path in label_files:
        has_minority = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.strip().split()[0])
                    if class_id == minority_class_id:
                        has_minority = 1
                        break
        y_stratify.append(has_minority)
    return np.array(y_stratify)

def setup_fold_directories(base_out_dir, fold_index):
    """
    Creates the necessary train and val directory structure for a specific fold.
    """
    fold_dir = os.path.join(base_out_dir, f"fold_{fold_index}")
    
    dirs = {
        'train_img': os.path.join(fold_dir, "train", "images"),
        'train_lbl': os.path.join(fold_dir, "train", "labels"),
        'val_img': os.path.join(fold_dir, "val", "images"),
        'val_lbl': os.path.join(fold_dir, "val", "labels")
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        
    return dirs, fold_dir

def copy_files_to_fold(indices, image_files, label_files, target_img_dir, target_lbl_dir, desc):
    """
    Copies image and label files to their designated fold directories.
    """
    for idx in tqdm(indices, desc=desc):
        img_src = image_files[idx]
        lbl_src = label_files[idx]
        
        # Copy image
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(target_img_dir, os.path.basename(img_src)))
            
        # Copy label (if it exists)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(target_lbl_dir, os.path.basename(lbl_src)))

def generate_yolo_yaml(fold_dir, class_names):
    """
    Creates a dataset.yaml file specifically for the current fold.
    """
    yaml_path = os.path.join(fold_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(os.path.join(fold_dir, 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(fold_dir, 'val'))}\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

def execute_stratified_kfold(src_img_dir, src_lbl_dir, base_out_dir, class_names, n_splits=5):
    """
    Main execution function to perform stratified K-Fold splitting.
    """
    # Gather all image files
    image_files = sorted(glob(os.path.join(src_img_dir, "*.jpg")) + glob(os.path.join(src_img_dir, "*.png")))
    
    # Map images to corresponding label files
    label_files = []
    for img in image_files:
        base_name = os.path.splitext(os.path.basename(img))[0]
        label_files.append(os.path.join(src_lbl_dir, f"{base_name}.txt"))
        
    image_files = np.array(image_files)
    label_files = np.array(label_files)
    
    # 0 represents 'slot_empty' in your dataset
    print("Scanning dataset for minority class distribution...")
    y_stratify = get_minority_class_distribution(label_files, minority_class_id=0)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold_idx, (train_index, val_index) in enumerate(skf.split(image_files, y_stratify), start=1):
        print(f"\n--- Processing Fold {fold_idx}/{n_splits} ---")
        
        dirs, fold_dir = setup_fold_directories(base_out_dir, fold_idx)
        
        copy_files_to_fold(train_index, image_files, label_files, dirs['train_img'], dirs['train_lbl'], f"Fold {fold_idx} - Train")
        copy_files_to_fold(val_index, image_files, label_files, dirs['val_img'], dirs['val_lbl'], f"Fold {fold_idx} - Val")
        
        generate_yolo_yaml(fold_dir, class_names)
        
    print("\n✅ Stratified 5-Fold splitting complete.")

if __name__ == "__main__":
    # Source directories (your combined dataset)
    SOURCE_IMAGES = "dataset/parking/all/images"
    SOURCE_LABELS = "dataset/parking/all/labels"
    
    # Output directory where the 5 folds will be saved
    OUTPUT_DIR = "dataset/dataset_kfold"
    
    # Class names exactly as they should appear in YOLO config
    CLASSES = ['slot_empty', 'slot_occ', 'space_free', 'no_park']
    
    execute_stratified_kfold(SOURCE_IMAGES, SOURCE_LABELS, OUTPUT_DIR, CLASSES)