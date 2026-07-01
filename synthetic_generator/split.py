import os
import shutil
import random

def split_dataset():
    base_dir = r"d:\Cogentic\sign-detection\synthetic_generator\synthetic_data"
    gt_file = os.path.join(base_dir, "rec_gt.txt")
    
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # FIX: Clear old split image folders to prevent stale images from previous
    # generations mixing in with newly generated ones.
    for split_dir in [train_dir, test_dir]:
        img_folder = os.path.join(split_dir, "images")
        if os.path.exists(img_folder):
            shutil.rmtree(img_folder)
            print(f"Cleared old images: {img_folder}")
    
    # Create fresh directories
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    
    # Read ground truth
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Clean up empty lines
    lines = [line for line in lines if line.strip()]
    
    # Group by base ID (syn_XXXXXX)
    from collections import defaultdict
    groups = defaultdict(list)
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            img_rel = parts[0].strip() # "images/syn_000000_1.png"
            filename = os.path.basename(img_rel)
            base_id = "_".join(filename.split("_")[:2]) # "syn_000000"
            groups[base_id].append(line)
            
    base_ids = list(groups.keys())
    
    # Shuffle base IDs
    random.seed(42)
    random.shuffle(base_ids)
    
    split_idx = int(len(base_ids) * 0.8)
    train_ids = base_ids[:split_idx]
    test_ids = base_ids[split_idx:]
    
    train_lines = []
    for bid in train_ids:
        train_lines.extend(groups[bid])
        
    test_lines = []
    for bid in test_ids:
        test_lines.extend(groups[bid])
    
    def process_split(split_lines, target_dir):
        gt_out = os.path.join(target_dir, "rec_gt.txt")
        with open(gt_out, "w", encoding="utf-8") as f:
            for line in split_lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_rel_path = parts[0].strip() # e.g. "images/syn_000003_1.png"
                    
                    # Original image path
                    src_img = os.path.join(base_dir, img_rel_path)
                    
                    # Target image path
                    dst_img = os.path.join(target_dir, img_rel_path)
                    
                    # Copy image
                    if os.path.exists(src_img):
                        shutil.copy2(src_img, dst_img)
                        f.write(line + "\n") # Write line to new gt
                    else:
                        print(f"Warning: Missing image {src_img}")

    print(f"Total samples: {len(lines)}")
    print(f"Moving {len(train_lines)} to train...")
    process_split(train_lines, train_dir)
    print(f"Moving {len(test_lines)} to test...")
    process_split(test_lines, test_dir)
    print("Done!")

if __name__ == "__main__":
    split_dataset()
