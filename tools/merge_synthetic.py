"""
tools/merge_synthetic.py
Merges synthetic split data into the real dataset ground-truth files.

Usage:
    python tools/merge_synthetic.py \
        --syn  synthetic_generator/synthetic_data \
        --real data/dataset
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Merge synthetic data into real dataset")
    p.add_argument("--syn",  required=True, help="Path to synthetic_data directory (contains train/ and test/)")
    p.add_argument("--real", required=True, help="Path to real dataset root (contains train/ and test/)")
    return p.parse_args()


def merge_split(syn_split: Path, real_split: Path):
    syn_gt  = syn_split / "rec_gt.txt"
    real_gt = real_split / "rec_gt.txt"
    real_img_dir = real_split / "crop_img"

    if not syn_gt.exists():
        print(f"  [SKIP] No synthetic GT found at {syn_gt}")
        return

    count = 0
    with open(syn_gt, "r", encoding="utf-8") as fs, \
         open(real_gt, "a", encoding="utf-8") as fr:
        for line in fs:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img_rel, label = parts[0].strip(), parts[1].strip()
            src = syn_split / img_rel
            if not src.exists():
                print(f"  [WARN] Missing: {src}")
                continue
            dst = real_img_dir / src.name
            shutil.copy2(src, dst)
            fr.write(f"crop_img/{src.name}\t{label}\n")
            count += 1

    print(f"  Merged {count} entries into {real_gt}")


def merge(args):
    syn_base  = Path(args.syn)
    real_base = Path(args.real)

    for split in ("train", "test"):
        print(f"Merging {split}...")
        merge_split(syn_base / split, real_base / split)

    print("Merge complete.")


if __name__ == "__main__":
    merge(parse_args())
