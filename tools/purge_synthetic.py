"""
tools/purge_synthetic.py
Removes synthetic entries from rec_gt.txt and deletes synthetic images.
Useful when you want to reset to real-data-only ground truth.

Usage:
    python tools/purge_synthetic.py --data data/dataset
"""

import argparse
import glob
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Purge synthetic data from dataset splits")
    p.add_argument("--data", required=True, help="Path to dataset root (contains train/ and test/)")
    return p.parse_args()


def clean_split(split_dir: Path):
    gt_file = split_dir / "rec_gt.txt"
    img_dir = split_dir / "crop_img"

    with open(gt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean = [l for l in lines if "syn_" not in l]

    with open(gt_file, "w", encoding="utf-8") as f:
        f.writelines(clean)

    deleted = 0
    for img in (img_dir).glob("syn_*.png"):
        img.unlink()
        deleted += 1

    print(f"  {split_dir.name}: kept {len(clean)} real lines, deleted {deleted} synthetic images.")


def purge(args):
    root = Path(args.data)
    for split in ("train", "test"):
        print(f"Cleaning {split}...")
        clean_split(root / split)
    print("Purge complete.")


if __name__ == "__main__":
    purge(parse_args())
