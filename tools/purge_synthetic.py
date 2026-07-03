"""
tools/purge_synthetic.py
Removes ALL synthetic entries from rec_gt.txt and deletes synthetic images.
Covers both the original generator (syn_*.png) and the alt-font generator
(altfont_*.png).  Useful when you want to reset to real-data-only ground truth.

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

    # Remove lines referencing any synthetic image (both generators)
    def _is_synthetic(line: str) -> bool:
        return "syn_" in line or "altfont_" in line

    clean = [l for l in lines if not _is_synthetic(l)]

    with open(gt_file, "w", encoding="utf-8") as f:
        f.writelines(clean)

    deleted = 0
    for pattern in ("syn_*.png", "altfont_*.png"):
        for img in img_dir.glob(pattern):
            img.unlink()
            deleted += 1

    removed_lines = len(lines) - len(clean)
    print(f"  {split_dir.name}: kept {len(clean)} real lines, "
          f"removed {removed_lines} synthetic GT lines, "
          f"deleted {deleted} synthetic images.")


def purge(args):
    root = Path(args.data)
    for split in ("train", "test"):
        print(f"Cleaning {split}...")
        clean_split(root / split)
    print("Purge complete.")


if __name__ == "__main__":
    purge(parse_args())
