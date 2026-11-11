import os
import shutil
import random
import math
import cv2
import torch
from glob import glob
from tqdm import tqdm

def load_image_as_tensor(image_path):
    """
    Reads an image and converts it to a normalized PyTorch tensor (C, H, W).
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor

def save_tensor_as_image(tensor, output_path):
    """
    Converts a PyTorch tensor back to an image and saves it to disk.
    """
    tensor = torch.clamp(tensor, 0.0, 1.0) * 255.0
    img_np = tensor.permute(1, 2, 0).byte().cpu().numpy()
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)

def apply_fds_linear_interpolation(target_tensor, source_tensor, alpha, beta=0.01):
    """
    Blends the low-frequency amplitude of the target and source tensors.
    """
    fft_target = torch.fft.fftshift(torch.fft.fft2(target_tensor, dim=(-2, -1), norm='ortho'))
    fft_source = torch.fft.fftshift(torch.fft.fft2(source_tensor, dim=(-2, -1), norm='ortho'))
    
    amp_target, phase_target = torch.abs(fft_target), torch.angle(fft_target)
    amp_source = torch.abs(fft_source)
    
    _, h, w = amp_target.shape
    b = math.floor(min(h, w) * beta)
    
    c_h, c_w = h // 2, w // 2
    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1
    
    amp_augmented = amp_target.clone()
    amp_augmented[:, h1:h2, w1:w2] = (1.0 - alpha) * amp_target[:, h1:h2, w1:w2] + alpha * amp_source[:, h1:h2, w1:w2]
    
    complex_spectrum = amp_augmented * torch.exp(1j * phase_target)
    fft_unshifted = torch.fft.ifftshift(complex_spectrum)
    reconstructed = torch.fft.ifft2(fft_unshifted, dim=(-2, -1), norm='ortho')
    
    return torch.real(reconstructed)

def setup_directory_structure(base_out_dir, fold_name):
    """
    Creates the train and val directory structure for the output dataset.
    """
    dirs = {
        'train_img': os.path.join(base_out_dir, fold_name, "train", "images"),
        'train_lbl': os.path.join(base_out_dir, fold_name, "train", "labels"),
        'val_img': os.path.join(base_out_dir, fold_name, "val", "images"),
        'val_lbl': os.path.join(base_out_dir, fold_name, "val", "labels")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def copy_unaltered_split(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, desc):
    """
    Directly copies images and labels without applying any FDS (used for val sets and baseline train images).
    """
    images = glob(os.path.join(src_img_dir, "*.*"))
    for img_path in tqdm(images, desc=desc, leave=False):
        base_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dst_img_dir, base_name))
        
        lbl_path = os.path.join(src_lbl_dir, os.path.splitext(base_name)[0] + ".txt")
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dst_lbl_dir, os.path.basename(lbl_path)))

def process_train_augmentation(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, ratio):
    """
    Generates 'ratio' number of FDS augmented images for each target image in the train set.
    """
    img_paths = sorted(glob(os.path.join(src_img_dir, "*.*")))
    
    for target_path in tqdm(img_paths, desc=f"Applying FDS (1:{ratio})", leave=False):
        target_tensor = load_image_as_tensor(target_path)
        base_name = os.path.splitext(os.path.basename(target_path))[0]
        ext = os.path.splitext(target_path)[1]
        target_lbl_path = os.path.join(src_lbl_dir, f"{base_name}.txt")
        
        for i in range(ratio):
            source_path = random.choice(img_paths)
            source_tensor = load_image_as_tensor(source_path)
            
            # Ensure tensors match in dimension for FFT operations
            if target_tensor.shape != source_tensor.shape:
                source_tensor = torch.nn.functional.interpolate(
                    source_tensor.unsqueeze(0), 
                    size=target_tensor.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            alpha = random.uniform(0.2, 0.8)
            aug_tensor = apply_fds_linear_interpolation(target_tensor, source_tensor, alpha)
            
            new_img_name = f"{base_name}_fds_{i}{ext}"
            save_tensor_as_image(aug_tensor, os.path.join(dst_img_dir, new_img_name))
            
            if os.path.exists(target_lbl_path):
                shutil.copy(target_lbl_path, os.path.join(dst_lbl_dir, f"{base_name}_fds_{i}.txt"))

def write_fold_yaml(yaml_src, out_fold_dir):
    """
    Writes dataset.yaml for an augmented fold: class definitions are taken from the
    baseline yaml, while the train/val paths point to the augmented fold's own folders.
    """
    with open(yaml_src, 'r') as f:
        lines = f.read().splitlines()
    out = []
    for line in lines:
        if line.startswith('train:'):
            line = f"train: {os.path.abspath(os.path.join(out_fold_dir, 'train'))}"
        elif line.startswith('val:'):
            line = f"val: {os.path.abspath(os.path.join(out_fold_dir, 'val'))}"
        out.append(line)
    with open(os.path.join(out_fold_dir, "dataset.yaml"), 'w') as f:
        f.write("\n".join(out) + "\n")

def build_fds_datasets(input_kfold_dir, ratios=[1, 3, 5], num_folds=5):
    """
    Main controller to loop through requested FDS ratios and folds.
    """
    for ratio in ratios:
        out_dataset_name = f"{input_kfold_dir}_FDS_1_{ratio}"
        print(f"\n======================================")
        print(f"Generating Dataset: {out_dataset_name}")
        print(f"======================================")
        
        for fold in range(1, num_folds + 1):
            fold_name = f"fold_{fold}"
            print(f"\nProcessing {fold_name}...")
            
            src_train_img = os.path.join(input_kfold_dir, fold_name, "train", "images")
            src_train_lbl = os.path.join(input_kfold_dir, fold_name, "train", "labels")
            src_val_img = os.path.join(input_kfold_dir, fold_name, "val", "images")
            src_val_lbl = os.path.join(input_kfold_dir, fold_name, "val", "labels")
            
            dirs = setup_directory_structure(out_dataset_name, fold_name)
            
            # 1. Copy unaltered validation set (CRITICAL: No data leakage)
            copy_unaltered_split(src_val_img, src_val_lbl, dirs['val_img'], dirs['val_lbl'], desc="Copying Val")
            
            # 2. Copy unaltered train baseline images
            copy_unaltered_split(src_train_img, src_train_lbl, dirs['train_img'], dirs['train_lbl'], desc="Copying Train Baseline")
            
            # 3. Generate the FDS augmented images
            process_train_augmentation(src_train_img, src_train_lbl, dirs['train_img'], dirs['train_lbl'], ratio)
            
            # 4. Write dataset.yaml pointing to THIS dataset's train/val folders.
            #    (Copying the baseline dataset.yaml would keep its absolute paths, so training
            #    would silently use the baseline images instead of the augmented ones.)
            yaml_src = os.path.join(input_kfold_dir, fold_name, "dataset.yaml")
            write_fold_yaml(yaml_src, os.path.join(out_dataset_name, fold_name))

if __name__ == "__main__":
    INPUT_KFOLD = "dataset/dataset_kfold"
    FDS_RATIOS = [1, 3, 5]
    
    build_fds_datasets(INPUT_KFOLD, FDS_RATIOS)