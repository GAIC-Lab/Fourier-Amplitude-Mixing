import os
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

def compute_dataset_statistics(label_dir, class_names):
    """
    Iterates through the dataset to accumulate instance counts and normalized areas.
    """
    label_files = glob(os.path.join(label_dir, "*.txt"))
    
    stats = {
        i: {'name': name, 'instances': 0, 'images_present': 0, 'total_normalized_area': 0.0} 
        for i, name in enumerate(class_names)
    }
    
    total_images = len(label_files)
    
    for filepath in tqdm(label_files, desc="Parsing label files"):
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

def print_statistics(stats, total_images):
    """
    Formats and prints the aggregated dataset statistics.
    """
    print(f"\nTotal Images Parsed: {total_images}\n")
    print(f"{'Class ID':<10} | {'Class Name':<15} | {'Instances':<10} | {'Images Present':<15} | {'Avg Area/Image (%)':<20}")
    print("-" * 75)
    
    for c_id, data in stats.items():
        name = data['name']
        instances = data['instances']
        img_present = data['images_present']
        total_area = data['total_normalized_area']
        
        # Calculates the average area footprint across the whole dataset
        avg_area_pct = (total_area / total_images) * 100 if total_images > 0 else 0
        
        print(f"{c_id:<10} | {name:<15} | {instances:<10} | {img_present:<15} | {avg_area_pct:.2f}%")

if __name__ == "__main__":
    # Update this path to the directory containing your YOLO .txt label files
    LABEL_DIR = "dataset/labels" 
    
    # Ensure the order matches your dataset.yaml
    CLASS_NAMES = ['slot_empty', 'slot_occ', 'space_free', 'no_park']
    
    computed_stats, total_imgs = compute_dataset_statistics(LABEL_DIR, CLASS_NAMES)
    print_statistics(computed_stats, total_imgs)