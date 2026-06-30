import os
import shutil
import random

def split_dataset():
    base_dir = r"d:\Cogentic\sign-detection\dataset_nived\dataset"
    gt_file = os.path.join(base_dir, "rec_gt.txt")
    
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # Create directories
    os.makedirs(os.path.join(train_dir, "crop_img"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "crop_img"), exist_ok=True)
    
    # Read ground truth
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Clean up empty lines
    lines = [line for line in lines if line.strip()]
    
    # Shuffle
    random.seed(42)
    random.shuffle(lines)
    
    split_idx = int(len(lines) * 0.8)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    
    def process_split(split_lines, target_dir):
        gt_out = os.path.join(target_dir, "rec_gt.txt")
        with open(gt_out, "w", encoding="utf-8") as f:
            for line in split_lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_rel_path = parts[0].strip()
                    text = parts[1].strip()
                    
                    # Original image path
                    src_img = os.path.join(base_dir, img_rel_path)
                    
                    # Target image path
                    dst_img = os.path.join(target_dir, img_rel_path)
                    
                    # Copy image
                    if os.path.exists(src_img):
                        shutil.copy2(src_img, dst_img)
                        f.write(line) # Write exact same line to new gt
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
