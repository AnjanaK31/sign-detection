"""
gen_altfont_test.py
-------------------
Generates synthetic TEST images using alternate fonts (Space Mono, Roboto Mono)
that were NOT used during training. Heavily weighted toward >=, <=, and % strings
which were previously missing from dict.txt.

Images are saved to:
    dataset_nived/dataset/test/crop_img/altfont/

Entries are APPENDED to:
    dataset_nived/dataset/test/rec_gt.txt

Usage (from repo root):
    python synthetic_generator/gen_altfont_test.py --n_texts 300 --n_aug 2
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
FONT_DIRS   = [
    REPO_ROOT / "synthetic_generator" / "fonts" / "Roboto_Mono,Space_Mono" / "Space_Mono",
    REPO_ROOT / "synthetic_generator" / "fonts" / "Roboto_Mono,Space_Mono" / "Roboto_Mono",
]
OUT_IMG_DIR = REPO_ROOT / "dataset_nived" / "dataset" / "test" / "crop_img" / "altfont"
LABEL_FILE  = REPO_ROOT / "dataset_nived" / "dataset" / "test" / "rec_gt.txt"

# ---------------------------------------------------------------------------
# Text generators (focused on failure characters)
# ---------------------------------------------------------------------------

def _val(lo, hi, decimals=None):
    dp = decimals if decimals is not None else random.choice([0, 1, 2])
    return round(random.uniform(lo, hi), dp)

def gen_gte():
    val  = _val(0.5, 200, 2)
    unit = random.choice(["", "%", "mm"])
    return random.choice([f">={val}{unit}", f"≥{val}{unit}"])

def gen_lte():
    val  = _val(0.5, 200, 2)
    unit = random.choice(["", "%", "mm", "µm"])
    return random.choice([f"<={val}{unit}", f"≤{val}{unit}"])

def gen_percentage():
    val    = _val(1, 100, random.choice([0, 1]))
    prefix = random.choice([">", "<", "≥", "≤", ""])
    return f"{prefix}{val}%"

def gen_greater_than():
    val  = _val(0.1, 200, random.choice([0, 1, 2]))
    unit = random.choice(["", "%", "mm", "Nm"])
    return f">{val}{unit}"

def gen_less_than():
    val  = _val(0.1, 600, random.choice([0, 1]))
    unit = random.choice(["", "dB", "µm", "mm", "%"])
    sep  = random.choice(["", " "])
    return f"<{val}{sep}{unit}".strip()

def gen_diameter():
    val = _val(1, 260, random.choice([0, 1, 2]))
    sep = random.choice(["", " "])
    return f"Ø{sep}{val}"

def gen_linear():
    val  = _val(0.5, 500, random.choice([0, 1, 2]))
    tol  = round(random.uniform(0.01, 1.5), random.choice([1, 2]))
    unit = random.choice(["mm", ""])
    return random.choice([f"{val}", f"{val} {unit}".strip(), f"{val}±{tol}"])

def gen_angle():
    val = _val(0, 360, random.choice([0, 1]))
    return f"{val}°"

def gen_plusminus():
    val = _val(0.1, 700, random.choice([0, 1, 2]))
    tol = round(random.uniform(0.1, 20), random.choice([1, 2]))
    return f"{val}±{tol}"

# Test corpus is biased toward the failure characters
GENERATORS = [
    (gen_gte,          25),
    (gen_lte,          25),
    (gen_percentage,   20),
    (gen_greater_than, 10),
    (gen_less_than,    10),
    (gen_diameter,      4),
    (gen_linear,        3),
    (gen_angle,         2),
    (gen_plusminus,     1),
]

def sample_text():
    fns, weights = zip(*GENERATORS)
    return random.choices(fns, weights=weights, k=1)[0]()

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def load_fonts(font_dirs, sizes):
    fonts = []
    for font_dir in font_dirs:
        font_dir = Path(font_dir)
        for fp in list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf")):
            for sz in sizes:
                try:
                    fonts.append(ImageFont.truetype(str(fp), sz))
                except Exception:
                    pass
    if not fonts:
        print("[warn] No fonts loaded, using PIL default.", file=sys.stderr)
        fonts = [ImageFont.load_default()]
    print(f"[info] Loaded {len(fonts)} font variants.")
    return fonts

def render_text(text, font, padding=6):
    dummy = Image.new("RGB", (1, 1))
    bbox  = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + padding * 2
    h = bbox[3] - bbox[1] + padding * 2
    if w < 4 or h < 4:
        return None
    img  = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=0)
    return np.array(img)

# ---------------------------------------------------------------------------
# Light augmentation (test images kept relatively clean)
# ---------------------------------------------------------------------------

def augment(img):
    noise = np.random.normal(0, random.uniform(1, 8), img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate alt-font test images for >=/<=/% evaluation")
    p.add_argument("--n_texts", type=int, default=300,  help="Unique text strings to generate")
    p.add_argument("--n_aug",   type=int, default=2,    help="Augmented variants per string")
    p.add_argument("--sizes",   default="24,32,40",     help="Comma-separated font sizes")
    p.add_argument("--seed",    type=int, default=77,   help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sizes = [int(s) for s in args.sizes.split(",")]
    fonts = load_fonts(FONT_DIRS, sizes)

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    corpus = [sample_text() for _ in range(args.n_texts)]
    print(f"Generated {len(corpus)} strings. Sample: {corpus[:6]}")

    new_lines = []
    skipped   = 0

    for i, text in enumerate(corpus):
        font = random.choice(fonts)
        base = render_text(text, font)
        if base is None:
            skipped += 1
            continue

        for j in range(args.n_aug):
            aug   = augment(base.copy())
            fname = f"syn_{i:06d}_{j}.png"
            cv2.imwrite(str(OUT_IMG_DIR / fname), aug)
            # Path relative to dataset/test/ — matches existing rec_gt.txt format
            new_lines.append(f"crop_img/altfont/{fname}	{text}")

    # Append to existing rec_gt.txt
    with open(LABEL_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(new_lines))

    print("\nDone.")
    print(f"  Images  : {len(new_lines)} -> {OUT_IMG_DIR}")
    print(f"  Appended: {len(new_lines)} lines -> {LABEL_FILE}")
    if skipped:
        print(f"  Skipped : {skipped} (font missing glyph?)")


if __name__ == "__main__":
    main()
