"""
tools/split_dataset.py
Splits the real crop dataset 80/20 into train and test splits.

Usage:
    python tools/split_dataset.py --data data/dataset
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split dataset into train/test (80/20)")
    p.add_argument("--data", required=True, help="Path to dataset root (contains crop_img/ and rec_gt.txt)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split(args):
    base    = Path(args.data)
    gt_file = base / "rec_gt.txt"

    with open(gt_file, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]

    random.seed(args.seed)
    random.shuffle(lines)

    split_idx   = int(len(lines) * 0.8)
    train_lines = lines[:split_idx]
    test_lines  = lines[split_idx:]

    for split_name, split_lines in [("train", train_lines), ("test", test_lines)]:
        split_dir = base / split_name
        (split_dir / "crop_img").mkdir(parents=True, exist_ok=True)

        gt_out = split_dir / "rec_gt.txt"
        with open(gt_out, "w", encoding="utf-8") as f:
            for line in split_lines:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_rel = parts[0].strip()
                    src     = base / img_rel
                    dst     = split_dir / img_rel
                    if src.exists():
                        shutil.copy2(src, dst)
                        f.write(line)
                    else:
                        print(f"  [WARN] Missing: {src}")

        print(f"  {split_name}: {len(split_lines)} samples")

    print(f"Split complete. Total: {len(lines)}")


if __name__ == "__main__":
    split(parse_args())
