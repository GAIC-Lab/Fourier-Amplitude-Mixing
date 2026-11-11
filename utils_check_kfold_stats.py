import os
import csv
from glob import glob
from tqdm import tqdm

def calculate_polygon_area(coords):
    """
    Calculates the normalized area of a polygon using the Shoelace formula.
    coords: List of floats [x1, y1, x2, y2, ..., xn, yn]
    """
    n = len(coords) // 2
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[2*i] * coords[2*j+1] - coords[2*j] * coords[2*i+1]
    return abs(area) / 2.0

def process_label_file(filepath):
    """
    Reads a single YOLO-seg format label file and extracts class IDs and areas.
    """
    instances = []
    if not os.path.exists(filepath):
        return instances
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                area = calculate_polygon_area(coords)
                instances.append({'class_id': class_id, 'area': area})
    return instances

def compute_split_statistics(label_dir, class_names):
    """
    Computes statistics for a single split folder (train or val).
    """
    label_files = glob(os.path.join(label_dir, "*.txt"))
    total_images = len(label_files)
    
    stats = {
        i: {'name': name, 'instances': 0, 'images_present': 0, 'total_normalized_area': 0.0} 
        for i, name in enumerate(class_names)
    }
    
    for filepath in label_files:
        instances = process_label_file(filepath)
        classes_in_image = set()
        
        for inst in instances:
            c_id = inst['class_id']
            if c_id in stats:
                stats[c_id]['instances'] += 1
                stats[c_id]['total_normalized_area'] += inst['area']
                classes_in_image.add(c_id)
                
        for c_id in classes_in_image:
            stats[c_id]['images_present'] += 1
            
    return stats, total_images

def analyze_all_folds(kfold_dir, class_names, num_folds=5):
    """
    Iterates through all folds and split directories to accumulate statistics.
    """
    all_rows = []
    
    for fold in range(1, num_folds + 1):
        fold_name = f"fold_{fold}"
        for split in ['train', 'val']:
            label_dir = os.path.join(kfold_dir, fold_name, split, "labels")
            
            print(f"Analyzing {fold_name} - {split}...")
            stats, total_imgs = compute_split_statistics(label_dir, class_names)
            
            for c_id, data in stats.items():
                name = data['name']
                instances = data['instances']
                img_present = data['images_present']
                total_area = data['total_normalized_area']
                
                img_pct = (img_present / total_imgs * 100) if total_imgs > 0 else 0.0
                avg_area_pct = (total_area / total_imgs * 100) if total_imgs > 0 else 0.0
                
                all_rows.append({
                    'Fold': fold_name,
                    'Split': split,
                    'Total_Images': total_imgs,
                    'Class_ID': c_id,
                    'Class_Name': name,
                    'Instances': instances,
                    'Images_Present': img_present,
                    'Images_Present_Pct': round(img_pct, 2),
                    'Avg_Area_Per_Image_Pct': round(avg_area_pct, 4)
                })
                
    return all_rows

def export_to_csv(stats_rows, output_csv_path):
    """
    Writes the statistical records into a CSV file.
    """
    fieldnames = [
        'Fold', 'Split', 'Total_Images', 'Class_ID', 'Class_Name', 
        'Instances', 'Images_Present', 'Images_Present_Pct', 'Avg_Area_Per_Image_Pct'
    ]
    
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
        
    print(f"\n✅ Statistics successfully exported to: {output_csv_path}")

if __name__ == "__main__":
    # Directory where dataset_kfold was saved
    KFOLD_DIR = "dataset/dataset_kfold"
    
    # CSV output file path
    OUTPUT_CSV = "kfold_dataset_stats.csv"
    
    CLASSES = ['slot_empty', 'slot_occ', 'space_free', 'no_park']
    
    results = analyze_all_folds(KFOLD_DIR, CLASSES)
    export_to_csv(results, OUTPUT_CSV)