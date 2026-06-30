import os
import shutil

def merge_split(syn_base, real_base, split_name):
    syn_gt_path = os.path.join(syn_base, split_name, "rec_gt.txt")
    real_gt_path = os.path.join(real_base, split_name, "rec_gt.txt")
    real_img_dir = os.path.join(real_base, split_name, "crop_img")
    syn_base_split = os.path.join(syn_base, split_name)
    
    if not os.path.exists(syn_gt_path):
        print(f"Skipping {split_name}, synthetic GT not found")
        return
        
    print(f"Merging {split_name}...")
    with open(syn_gt_path, 'r', encoding='utf-8') as f_syn, open(real_gt_path, 'a', encoding='utf-8') as f_real:
        for line in f_syn:
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                img_rel = parts[0] # "images/syn_xxx.png"
                label = parts[1]
                
                # Copy image
                src_img = os.path.join(syn_base_split, img_rel)
                filename = os.path.basename(img_rel)
                dst_img = os.path.join(real_img_dir, filename)
                
                if os.path.exists(src_img):
                    shutil.copy2(src_img, dst_img)
                    # Append to real gt
                    f_real.write(f"crop_img/{filename}\t{label}\n")
                else:
                    print(f"Warning: {src_img} not found")

if __name__ == "__main__":
    syn_base = r"d:\Cogentic\sign-detection\synthetic_generator\synthetic_data"
    real_base = r"d:\Cogentic\sign-detection\dataset_nived\dataset"
    
    merge_split(syn_base, real_base, "train")
    merge_split(syn_base, real_base, "test")
    print("Merge complete!")
