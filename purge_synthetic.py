import os
import glob

def clean_split(split_dir):
    gt_file = os.path.join(split_dir, "rec_gt.txt")
    img_dir = os.path.join(split_dir, "crop_img")
    
    # 1. Clean rec_gt.txt
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    clean_lines = [line for line in lines if "syn_" not in line]
    
    with open(gt_file, "w", encoding="utf-8") as f:
        f.writelines(clean_lines)
        
    # 2. Delete synthetic images
    syn_imgs = glob.glob(os.path.join(img_dir, "syn_*.png"))
    for img in syn_imgs:
        os.remove(img)
        
    print(f"Cleaned {split_dir}: kept {len(clean_lines)} real lines, deleted {len(syn_imgs)} synthetic images.")

base = r"d:\Cogentic\sign-detection\dataset_nived\dataset"
clean_split(os.path.join(base, "train"))
clean_split(os.path.join(base, "test"))
