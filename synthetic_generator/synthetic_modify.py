"""
synth_cad_augment.py
--------------------
Generates synthetic training images from real CAD text crops.

Usage:
    python synth_cad_augment.py \
        --input_dir  /path/to/real_crops \
        --output_dir /path/to/synth_output \
        --label_file labels.txt \
        --per_image  50

Each real crop + its ground-truth label → N augmented variants.
Output: flat folder of PNGs  +  a PaddleOCR-compatible label file
        (tab-separated: relative_path\tlabel)
"""

import argparse, os, random, math, textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL AUGMENTATION PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── 1. Noise ───────────────────────────────────────────────────────────────────

def add_gaussian_noise(gray, sigma_range=(2, 5)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, gray.shape).astype(np.float32)
    out = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def add_salt_pepper(gray, amount_range=(0.002, 0.015)):
    amount = random.uniform(*amount_range)
    out = gray.copy()
    n_salt = int(gray.size * amount * 0.5)
    coords = [np.random.randint(0, i, n_salt) for i in gray.shape]
    out[tuple(coords)] = 255
    coords = [np.random.randint(0, i, n_salt) for i in gray.shape]
    out[tuple(coords)] = 0
    return out


# ── 2. Blur / sharpness ────────────────────────────────────────────────────────

def random_blur(gray, k_range=(1, 3)):
    k = random.choice(range(k_range[0], k_range[1]+1, 2)) | 1   # must be odd
    return cv2.GaussianBlur(gray, (k, k), 0)

def random_sharpen(pil_img):
    factor = random.uniform(1.2, 2.5)
    return ImageEnhance.Sharpness(pil_img).enhance(factor)


# ── 3. Brightness / contrast ───────────────────────────────────────────────────

def random_brightness(pil_img, lo=0.6, hi=1.4):
    return ImageEnhance.Brightness(pil_img).enhance(random.uniform(lo, hi))

def random_contrast(pil_img, lo=0.7, hi=1.5):
    return ImageEnhance.Contrast(pil_img).enhance(random.uniform(lo, hi))


# ── 4. Geometric ───────────────────────────────────────────────────────────────

def random_rotate(gray, max_deg=3.0):
    """Small rotation (perspective skew in real drawings)."""
    angle = random.uniform(-max_deg, max_deg)
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    bg = int(np.median(gray))          # fill with background tone
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg)

def random_shear(gray, max_shear=0.08):
    """Horizontal shear — simulates italic/oblique text."""
    h, w = gray.shape
    sx = random.uniform(-max_shear, max_shear)
    M = np.float32([[1, sx, 0], [0, 1, 0]])
    new_w = int(w + abs(sx) * h)
    bg = int(np.median(gray))
    out = cv2.warpAffine(gray, M, (new_w, h),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=bg)
    # crop back to original width
    if sx >= 0:
        return out[:, :w]
    else:
        return out[:, new_w - w:]

def random_scale(gray, lo=0.80, hi=1.25):
    """Uniform scale then crop/pad back to original size."""
    h, w = gray.shape
    factor = random.uniform(lo, hi)
    new_h, new_w = max(1, int(h * factor)), max(1, int(w * factor))
    scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bg = int(np.median(gray))
    # centre-crop or centre-pad
    canvas = np.full((h, w), bg, dtype=np.uint8)
    y0 = max(0, (new_h - h) // 2)
    x0 = max(0, (new_w - w) // 2)
    cy0 = max(0, (h - new_h) // 2)
    cx0 = max(0, (w - new_w) // 2)
    rh = min(new_h, h) - max(0, (new_h - h) // 2) - max(0, (h - new_h) // 2)
    rw = min(new_w, w) - max(0, (new_w - w) // 2) - max(0, (w - new_w) // 2)
    rh = min(rh, h - cy0, new_h - y0)
    rw = min(rw, w - cx0, new_w - x0)
    if rh > 0 and rw > 0:
        canvas[cy0:cy0+rh, cx0:cx0+rw] = scaled[y0:y0+rh, x0:x0+rw]
    return canvas

def random_perspective(gray, strength=0.04):
    """Mild four-point perspective warp."""
    h, w = gray.shape
    d = strength
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    def jitter():
        return random.uniform(-d * min(h, w), d * min(h, w))
    dst_pts = src_pts + np.float32([[jitter(), jitter()] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bg = int(np.median(gray))
    return cv2.warpPerspective(gray, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=bg)


# ── 5. Degradation (scanner/print artefacts) ───────────────────────────────────

def random_jpeg_artefact(gray, quality_range=(40, 85)):
    quality = random.randint(*quality_range)
    _, enc = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def add_ink_bleed(gray, iterations_range=(0, 1)):
    """Dilate dark pixels to simulate ink bleed-through."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    n = random.randint(*iterations_range)
    # dark-on-light: invert → dilate → invert
    inv = 255 - gray
    dilated = cv2.dilate(inv, k, iterations=n)
    return 255 - dilated

def add_erosion(gray, iterations_range=(1, 2)):
    """Erode dark strokes — thin pen effect."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    n = random.randint(*iterations_range)
    inv = 255 - gray
    eroded = cv2.erode(inv, k, iterations=n)
    return 255 - eroded

def add_scan_lines(gray, alpha_range=(0.05, 0.20)):
    """Faint horizontal banding (flatbed scanner)."""
    out = gray.astype(np.float32)
    alpha = random.uniform(*alpha_range)
    for y in range(0, gray.shape[0], random.randint(3, 8)):
        out[y] = np.clip(out[y] * (1 - alpha), 0, 255)
    return out.astype(np.uint8)

def add_background_texture(gray, intensity_range=(5, 25)):
    """Low-frequency noise background (aged paper)."""
    h, w = gray.shape
    intensity = random.uniform(*intensity_range)
    # coarse noise upsampled
    small = np.random.uniform(0, intensity, (max(1, h // 8), max(1, w // 8)))
    texture = cv2.resize(small.astype(np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC)
    out = np.clip(gray.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    return out

def add_line_artifact(gray, n_lines_range=(1, 3)):
    """Random thin horizontal/vertical lines (dimension lines bleeding in)."""
    out = gray.copy()
    n = random.randint(*n_lines_range)
    h, w = gray.shape
    for _ in range(n):
        if random.random() < 0.6:   # horizontal
            y = random.randint(0, h - 1)
            x1, x2 = sorted(random.sample(range(w), 2))
            thickness = random.randint(1, 2)
            color = random.randint(0, 80)
            cv2.line(out, (x1, y), (x2, y), color, thickness)
        else:                        # vertical
            x = random.randint(0, w - 1)
            y1, y2 = sorted(random.sample(range(h), 2))
            thickness = random.randint(1, 2)
            color = random.randint(0, 80)
            cv2.line(out, (x, y1), (x, y2), color, thickness)
    return out


# ── 6. Padding / crop jitter ───────────────────────────────────────────────────

def random_pad(gray, pad_range=(2, 12)):
    """Add random white border (crop from scanner over-scan)."""
    h, w = gray.shape
    bg = int(np.percentile(gray, 95))   # near-white
    pad = random.randint(*pad_range)
    pt = random.randint(0, pad)
    pb = random.randint(0, pad)
    pl = random.randint(0, pad)
    pr = random.randint(0, pad)
    out = cv2.copyMakeBorder(gray, pt, pb, pl, pr,
                             cv2.BORDER_CONSTANT, value=bg)
    return out

def random_crop_jitter(gray, jitter=4):
    """Tiny random crop to simulate imprecise bounding-box extraction."""
    h, w = gray.shape
    if h < jitter * 2 + 4 or w < jitter * 2 + 4:
        return gray
    t = random.randint(0, jitter)
    b = random.randint(0, jitter)
    l = random.randint(0, jitter)
    r = random.randint(0, jitter)
    return gray[t:h-b-1, l:w-r-1]


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

AUGMENT_GROUPS = [
    # (name, probability, fn)
    ("noise_gauss",   0.60, add_gaussian_noise),
    ("noise_sp",      0.30, add_salt_pepper),
    ("blur",          0.50, random_blur),
    # ("rotate",        0.70, random_rotate),
    ("shear",         0.40, random_shear),
    # ("scale",         0.50, random_scale),
    ("perspective",   0.35, random_perspective),
    ("jpeg",          0.45, random_jpeg_artefact),
    ("scan_lines",    0.20, add_scan_lines),
    ("bg_texture",    0.40, add_background_texture),
    # ("line_artifact", 0.25, add_line_artifact),
    ("pad",           0.50, random_pad),
    # ("crop_jitter",   0.40, random_crop_jitter),

    # ("erosion",       0.20, add_erosion),
    # ("ink_bleed",     0.25, add_ink_bleed),
]

PIL_AUGMENTS = [
    ("brightness",  0.50, random_brightness),
    ("contrast",    0.50, random_contrast),
    ("sharpen",     0.30, random_sharpen),
]


def augment_once(gray: np.ndarray) -> np.ndarray:
    """Apply a random subset of augmentations and return grayscale uint8."""
    out = gray.copy()

    # random subset of cv2-based ops
    for _name, prob, fn in AUGMENT_GROUPS:
        if random.random() < prob:
            try:
                candidate = fn(out)
                if candidate is not None and candidate.size > 0:
                    out = candidate
            except Exception:
                pass   # skip silently

    # PIL-based ops
    pil = Image.fromarray(out)
    for _name, prob, fn in PIL_AUGMENTS:
        if random.random() < prob:
            try:
                pil = fn(pil)
            except Exception:
                pass
    out = np.array(pil.convert("L"))

    # ensure minimum size
    h, w = out.shape
    if h < 8 or w < 8:
        out = gray.copy()

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(label_file: str) -> dict:
    """
    Expect tab-separated file: filename<TAB>label
    e.g.   msil_crop_59.jpg   Ø7.4
    Returns {stem: label}
    """
    labels = {}
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                fname, label = parts
                labels[Path(fname).stem] = label
    return labels


def generate(input_dir: str, output_dir: str, label_file: str,
             per_image: int = 50):

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(label_file)

    manifest_lines = []

    image_files = sorted(input_dir.glob("*.jpg")) + \
                  sorted(input_dir.glob("*.png")) + \
                  sorted(input_dir.glob("*.jpeg"))

    if not image_files:
        print("No images found in", input_dir)
        return

    total = 0
    for img_path in image_files:
        stem  = img_path.stem
        label = labels.get(stem)
        if label is None:
            print(f"  [SKIP] no label for {img_path.name}")
            continue

        bgr  = cv2.imread(str(img_path))
        gray = to_gray(bgr)

        for i in range(per_image):
            aug = augment_once(gray)

            out_name = f"{stem}_aug{i:04d}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), aug)

            manifest_lines.append(f"{out_name}\t{label}")
            total += 1

        print(f"  {img_path.name!r:30s} → {per_image} variants  label={label!r}")

    # write PaddleOCR-style label file
    manifest_path = output_dir / "rec_gt.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"\nDone. {total} images written → {output_dir}")
    print(f"Label file: {manifest_path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CAD OCR synthetic training data generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Label file format (tab-separated):
              msil_crop_59.jpg    Ø7.4
              msil_crop_107.jpg   2×Ø8
              polaris_crop_41.jpg Ø59.4 MAX
        """)
    )
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--per_image",  type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    generate(args.input_dir, args.output_dir, args.label_file, args.per_image)
