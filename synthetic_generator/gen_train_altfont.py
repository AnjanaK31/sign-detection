"""
gen_train_altfont.py
--------------------
Generates synthetic TRAINING images using alternate fonts (Space Mono, Roboto Mono)
that were NOT seen during the original training run.

Two primary focuses:
  1. Ø (diameter) — weight ~38% of generated corpus     (under-detected in real images)
  2. Numbers & letters in new fonts — weight ~35%       (model failed on new-font alphanumerics)

Images are saved to:
    synthetic_generator/synthetic_data/train/images/altfont/

Entries are APPENDED to:
    synthetic_generator/synthetic_data/train/rec_gt.txt

Usage (from repo root):
    python synthetic_generator/gen_train_altfont.py
    python synthetic_generator/gen_train_altfont.py --n_texts 3000 --n_aug 5 --seed 99
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
OUT_IMG_DIR = REPO_ROOT / "synthetic_generator" / "synthetic_data" / "train" / "images" / "altfont"
LABEL_FILE  = REPO_ROOT / "synthetic_generator" / "synthetic_data" / "train" / "rec_gt.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(lo, hi, decimals=None):
    dp = decimals if decimals is not None else random.choice([0, 1, 2])
    return round(random.uniform(lo, hi), dp)

def _tol(lo=0.01, hi=1.5):
    return round(random.uniform(lo, hi), random.choice([1, 2]))

# ---------------------------------------------------------------------------
# Text generators
# ---------------------------------------------------------------------------

# ── PRIMARY: Ø (diameter) — boosted ──────────────────────────────────────────
def gen_diameter():
    val = _val(1, 260, random.choice([0, 1, 2]))
    tol = _tol()
    sep = random.choice(["", " "])
    return random.choice([
        f"O{sep}{val}",
        f"O{sep}{val}mm",
        f"O{sep}{val}+{tol}",
        f"O{sep}{val} +{tol}",
        f"O{sep}{val} MAX",
        f"O{sep}{val} MIN",
        f"O{sep}{val} (SPHER)",
        f"O{sep}{val} REF",
        f"O{sep}{val}+{round(random.uniform(0.01, 0.5), 2)}/-0",
    ]).replace("O", "Ø")  # use Ø explicitly

def gen_diameter_qty():
    """e.g. 4X Ø12.5 — quantity + diameter pattern."""
    n      = random.randint(2, 8)
    val    = _val(1, 100, random.choice([0, 1, 2]))
    sep    = random.choice(["", " "])
    qty_sep = random.choice(["X ", "x "])
    return f"{n}{qty_sep}Ø{sep}{val}"

# ── SECONDARY: Alphanumerics in new font ──────────────────────────────────────
def gen_pure_number():
    val  = _val(0.01, 9999, random.choice([0, 1, 2]))
    unit = random.choice(["", "mm", "%", "°", " mm", " %"])
    return f"{val}{unit}"

def gen_pure_letters():
    return random.choice([
        "MAX", "MIN", "REF", "TYP", "SPHER", "NOTES", "SCALE",
        "MATERIAL", "STEEL", "FINISH", "SECTION", "DETAIL",
        "REMOVE", "BURRS", "QTY", "SURFACE", "ALL", "DIMS",
        "SEE", "NOTE", "TOLERANCE", "DATUM", "BASIC",
        "APPROX", "REQUIRED", "TYPICAL", "UNLESS", "NOTED",
    ])

def gen_part_number():
    prefix = random.choice(["PN ", "P/N ", "PART NO. ", ""])
    num    = "".join(random.choices("0123456789", k=random.randint(3, 7)))
    suffix = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 3)))
    sep    = random.choice(["-", "", " "])
    return f"{prefix}{num}{sep}{suffix}"

def gen_mixed_alphanum():
    chars = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=random.randint(4, 8)))
    return random.choice([
        chars,
        f"REV {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1, 9)}",
        f"DWG {''.join(random.choices('0123456789', k=6))}",
        f"{''.join(random.choices('0123456789', k=3))}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}",
        f"{random.randint(1,9)}X {''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{random.randint(10, 99)}",
    ])

def gen_cad_note():
    return random.choice([
        "SEE DETAIL A", "ALL DIMS IN MM", "MATERIAL: STEEL",
        "SCALE 1:1", "SECTION B-B", "REMOVE BURRS",
        "TOLERANCE ±0.1", "QTY: 4", "SURFACE FINISH",
        "DO NOT SCALE", "UNLESS OTHERWISE NOTED",
        "BREAK SHARP EDGES", "GENERAL TOLERANCE",
        "ALL DIMS IN MM UNLESS NOTED",
    ])

# ── Standard CAD (maintain coverage) ─────────────────────────────────────────
def gen_plusminus():
    return f"{_val(0.1, 700, random.choice([0, 1, 2]))}±{_tol(0.1, 20)}"

def gen_angle():
    val = _val(0, 360, random.choice([0, 1]))
    tol = _val(0.5, 10, random.choice([0, 1]))
    return random.choice([f"{val}°", f"{val}°±{tol}°", f"{val}°±{tol}"])

def gen_gte():
    val  = _val(0.5, 200, 2)
    unit = random.choice(["", "%", "mm"])
    return random.choice([f">={val}{unit}", f"≥{val}{unit}"])

def gen_lte():
    val  = _val(0.5, 200, 2)
    unit = random.choice(["", "%", "mm", "µm"])
    return random.choice([f"<={val}{unit}", f"≤{val}{unit}"])

def gen_percentage():
    val = _val(1, 100, random.choice([0, 1]))
    return f"{random.choice(['>', '<', '≥', '≤', ''])}{val}%"

def gen_thread():
    d   = random.choice([3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24])
    p   = random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    tc  = random.choice(["6H", "6g", "6f", "5H", "7H", ""])
    ps  = random.choice(["x", "X", "×"])
    tc_str = f"-{tc}" if tc else ""
    qty = random.choice(["", f"{random.randint(2,8)}X ", f"{random.randint(2,8)}× "])
    return f"{qty}M{d}{ps}{p}{tc_str}"

def gen_linear():
    val  = _val(0.5, 500, random.choice([0, 1, 2]))
    unit = random.choice(["mm", ""])
    return random.choice([
        f"{val}", f"{val} {unit}".strip(),
        f"{val}±{_tol()}", f"{val}±{_tol()} {unit}".strip(),
    ])

def gen_greater_than():
    val = _val(0.1, 200, random.choice([0, 1, 2]))
    return f">{val}{random.choice(['', '%', 'mm', 'Nm'])}"

def gen_less_than():
    val  = _val(0.1, 600, random.choice([0, 1]))
    unit = random.choice(["", "dB", "µm", "mm", "%"])
    return f"<{val}{random.choice(['', ' '])}{unit}".strip()

def gen_micrometre():
    val = _val(0.1, 1000, random.choice([0, 1]))
    return f"{random.choice(['', '<', '>', '±'])}{val}{random.choice(['', ' '])}µm"

def gen_radius():
    val = _val(0.5, 150, random.choice([0, 1]))
    sep = random.choice(["", " "])
    return random.choice([f"R{sep}{val}mm", f"R{sep}{val}"])

def gen_parentheses():
    val = _val(1, 100, random.choice([0, 1]))
    return random.choice([f"({val})", f"({val} REF)", f"({val} MAX)", "(TYP)", "(SEE NOTE)"])

# ---------------------------------------------------------------------------
# Generator table
# Ø diameter ~38% | alphanumeric ~35% | standard CAD ~27%
# ---------------------------------------------------------------------------
GENERATORS = [
    # PRIMARY: Ø diameter
    (gen_diameter,       40),
    (gen_diameter_qty,   20),
    # SECONDARY: alphanumeric (new-font coverage)
    (gen_pure_number,    20),
    (gen_pure_letters,   15),
    (gen_part_number,    15),
    (gen_mixed_alphanum, 10),
    (gen_cad_note,       10),
    # Standard CAD (maintain existing coverage)
    (gen_plusminus,       5),
    (gen_angle,           8),
    (gen_gte,             8),
    (gen_lte,             8),
    (gen_percentage,      6),
    (gen_thread,          4),
    (gen_linear,          4),
    (gen_greater_than,    4),
    (gen_less_than,       4),
    (gen_micrometre,      3),
    (gen_radius,          3),
    (gen_parentheses,     3),
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
        print("[warn] No fonts loaded — falling back to PIL default.", file=sys.stderr)
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
# Augmentation
# ---------------------------------------------------------------------------

def aug_noise(img):
    noise = np.random.normal(0, random.uniform(2, 15), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def aug_blur(img):
    return cv2.GaussianBlur(img, (random.choice([3, 5]),) * 2, 0)

def aug_rotate(img):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-4, 4), 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)

def aug_scale(img):
    h, w = img.shape
    scale = random.uniform(0.90, 1.30)
    nh, nw = max(4, int(h * scale)), max(4, int(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

def aug_brightness(img):
    return np.clip(img.astype(np.int16) + random.randint(-40, 40), 0, 255).astype(np.uint8)

def aug_jpeg(img):
    q = random.randint(35, 80)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def aug_perspective(img):
    h, w = img.shape
    d = random.randint(1, min(5, h // 4, w // 4))
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), random.randint(0, d)],
        [random.randint(0, d), h - random.randint(0, d)],
        [w - random.randint(0, d), h - random.randint(0, d)],
    ])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=255)

def aug_dilate_erode(img):
    k = np.ones((2, 2), np.uint8)
    return cv2.dilate(img, k) if random.random() < 0.5 else cv2.erode(img, k)

def aug_thin_lines(img):
    out = img.copy()
    h, w = out.shape
    for _ in range(random.randint(1, 3)):
        if random.random() < 0.5:
            x = random.randint(0, w - 1)
            cv2.line(out, (x, 0), (x, h - 1), random.randint(100, 200), 1)
        else:
            y = random.randint(0, h - 1)
            cv2.line(out, (0, y), (w - 1, y), random.randint(100, 200), 1)
    return out

AUGMENTATIONS = [
    (aug_noise,        4),
    (aug_blur,         3),
    (aug_rotate,       5),
    (aug_scale,        12),
    (aug_brightness,   3),
    (aug_jpeg,         4),
    (aug_thin_lines,   3),
    (aug_perspective,  3),
    (aug_dilate_erode, 2),
]

def augment(img, n=3):
    fns, weights = zip(*AUGMENTATIONS)
    for fn in random.choices(fns, weights=weights, k=n):
        try:
            img = fn(img)
        except Exception:
            pass
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate alt-font TRAINING images (Ø boosted + alphanumeric focus)"
    )
    p.add_argument("--n_texts", type=int, default=2000,
                   help="Unique text strings to generate (default: 2000)")
    p.add_argument("--n_aug",   type=int, default=4,
                   help="Augmented variants per string (default: 4)")
    p.add_argument("--sizes",   default="24,32,40,48",
                   help="Comma-separated font sizes")
    p.add_argument("--seed",    type=int, default=77,
                   help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sizes  = [int(s) for s in args.sizes.split(",")]
    fonts  = load_fonts(FONT_DIRS, sizes)

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    corpus      = [sample_text() for _ in range(args.n_texts)]
    diam_count  = sum(1 for t in corpus if "Ø" in t)
    total_imgs  = len(corpus) * args.n_aug

    print(f"[info] Generated {len(corpus)} strings.")
    print(f"  Ø strings  : {diam_count}  ({diam_count/len(corpus)*100:.1f}%)")
    print(f"  Total imgs : {total_imgs}  ({len(corpus)} texts × {args.n_aug} aug)")
    print(f"  Sample     : {corpus[:6]}")

    new_lines = []
    skipped   = 0

    for i, text in enumerate(corpus):
        font = random.choice(fonts)
        base = render_text(text, font)
        if base is None:
            skipped += 1
            continue

        for j in range(args.n_aug):
            aug   = augment(base.copy(), n=random.randint(2, 4))
            fname = f"altfont_{i:06d}_{j}.png"
            cv2.imwrite(str(OUT_IMG_DIR / fname), aug)
            # Path relative to synthetic_data/train/ — matches existing rec_gt.txt format
            new_lines.append(f"images/altfont/{fname}\t{text}")

    # Append to existing training rec_gt.txt (never overwrites)
    with open(LABEL_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(new_lines) + "\n")

    print(f"\nDone.")
    print(f"  Images  : {len(new_lines)}  ->  {OUT_IMG_DIR}")
    print(f"  Appended: {len(new_lines)} lines  ->  {LABEL_FILE}")
    if skipped:
        print(f"  Skipped : {skipped} (font missing glyph?)")


if __name__ == "__main__":
    main()
