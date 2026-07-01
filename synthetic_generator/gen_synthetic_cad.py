"""
gen_synthetic_cad.py
Synthetic training data generator for CAD text recognition (PP-OCRv6 format).

Covers all symbol classes observed in real drawings:
  ±  °  Ø  R  SPH R  >  <  >=  "  µm
  M (thread)  MAX  MIN  REF  limit-tolerance  TYP  nX  ×  *

Usage:
    python gen_synthetic_cad.py \
        --out_dir ./synthetic_data \
        --fonts /path/to/fonts \
        --real_bg_dir ./real_cad_crops \   # optional: dir of grayscale bg patches
        --n_texts 5000 \
        --n_aug 5
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 1. Text corpus generators
# Each function returns one realistic CAD annotation string.
# Weights in GENERATORS reflect observed frequency in real drawings.
# ---------------------------------------------------------------------------

def _val(lo: float, hi: float, decimals: int | None = None) -> float:
    """Random float in [lo, hi] with 0–2 decimal places."""
    dp = decimals if decimals is not None else random.choice([0, 1, 2])
    return round(random.uniform(lo, hi), dp)

def _tol(lo: float = 0.01, hi: float = 1.5) -> float:
    return round(random.uniform(lo, hi), random.choice([1, 2]))

# ── Ø  Diameter ──────────────────────────────────────────────────────────────
def gen_diameter() -> str:
    # real examples: Ø26.99mm, Ø242, Ø23.81mm, Ø232.0 MAX, Ø10.05+0.05, Ø 245
    val   = _val(1, 260, random.choice([0, 1, 2]))
    tol   = _tol()
    sep   = random.choice(["", " "])   # FIX: Ø 245 (spaced) vs Ø245
    forms = [
        f"Ø{sep}{val}",
        f"Ø{sep}{val}±{tol}",
        f"Ø{sep}{val} ±{tol}",
        f"Ø{sep}{val} MAX",
        f"Ø{sep}{val} MIN",
        f"Ø{sep}{val} (SPHER)",         # FIX: Ø9.9 (SPHER) pattern
        f"Ø{sep}{val}.{random.randint(0,9)} (SPHER)",
    ]
    return random.choice(forms)

# ── ±  Plus/Minus tolerance ──────────────────────────────────────────────────
def gen_plusminus() -> str:
    val = _val(0.1, 700, random.choice([0, 1, 2]))
    tol = _tol(0.1, 20)
    unit = random.choice(["", "mm"])
    sep  = random.choice(["", " "])
    return f"{val}±{tol}{sep}{unit}".strip()

# ── R  Radius ─────────────────────────────────────────────────────────────────
def gen_radius() -> str:
    # real: R90mm, R19.5, SPH R (handled separately)
    val = _val(0.5, 150, random.choice([0, 1]))
    unit = random.choice(["mm", ""])
    sep  = random.choice(["", " "])
    return random.choice([f"R{sep}{val}{unit}", f"R{sep}{val}"])

# ── SPH R  Spherical Radius ───────────────────────────────────────────────────
def gen_sph_radius() -> str:
    val = _val(0.5, 100, random.choice([0, 1]))
    return random.choice([f"SPH R{val}", f"SPH R {val}mm"])

# ── °  Angle ──────────────────────────────────────────────────────────────────
def gen_angle() -> str:
    # real: 18.1 deg, 90°, 15°, 10 deg, 35° ±3°
    val = _val(0, 360, random.choice([0, 1]))
    tol = _val(0.5, 10, random.choice([0, 1]))
    sep = random.choice(["", " "])
    return random.choice([
        f"{val}°",
        f"{val}°{sep}±{tol}°",           # FIX: 35° ±3° toleranced angle
        f"{val}°{sep}±{tol}",
    ])

# ── >  Greater-than ───────────────────────────────────────────────────────────
def gen_greater_than() -> str:
    # real: >40%, >98%, >1.33
    val = _val(0.1, 200, random.choice([0, 1, 2]))
    unit = random.choice(["", "%", "mm", "Nm"])
    return f">{val}{unit}"

# ── <  Less-than ──────────────────────────────────────────────────────────────
def gen_less_than() -> str:
    # real: <25 PPM, <30dB, <500 µm
    val = _val(0.1, 600, random.choice([0, 1]))
    unit = random.choice(["", "dB", "µm", "mm"])
    sep  = random.choice(["", " "])
    return f"<{val}{sep}{unit}".strip()

# ── >=  Greater-than-or-equal ─────────────────────────────────────────────────
def gen_gte() -> str:
    # real: >=1.67, ≥98%
    val = _val(0.5, 200, 2)
    unit = random.choice(["", "%", "mm"])
    return random.choice([f">={val}{unit}", f"≥{val}{unit}"])

# ── <=  Less-than-or-equal ────────────────────────────────────────────────────
def gen_lte() -> str:
    # mirror of gen_gte: <=0.05, ≤500 µm
    val = _val(0.5, 200, 2)
    unit = random.choice(["", "%", "mm", "µm"])
    return random.choice([f"<={val}{unit}", f"≤{val}{unit}"])

# ── %  Percentage ─────────────────────────────────────────────────────────────
def gen_percentage() -> str:
    # real: >40%, <98%, ≥95%, efficiency specs in CAD notes
    val = _val(1, 100, random.choice([0, 1]))
    prefix = random.choice([">" , "<", "≥", "≤", ""])
    return f"{prefix}{val}%"

# ── "  Inches ─────────────────────────────────────────────────────────────────
def gen_inches() -> str:
    # real: 10" (booster size)
    val = _val(0.25, 24, random.choice([0, 1, 2]))
    return f'{val}"'

# ── µm  Micrometre ────────────────────────────────────────────────────────────
def gen_micrometre() -> str:
    # real: 500 µm, <500 µm
    val = _val(0.1, 1000, random.choice([0, 1]))
    prefix = random.choice(["", "<", ">", "±"])
    sep    = random.choice(["", " "])
    return f"{prefix}{val}{sep}µm"

# ── M  Metric thread ──────────────────────────────────────────────────────────
def gen_thread() -> str:
    # real: M12x1-6H, M12×1 mm JASO, M10X1-6H (TYP), 4X M8 x1.25, 4×M8×1.25
    pitches      = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    diameters    = [3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24]
    tol_classes  = ["6H", "6g", "6f", "5H", "7H", "6H", ""]
    pitch_seps   = ["x", "X", "×"]
    d = random.choice(diameters)
    p = random.choice(pitches)
    tc = random.choice(tol_classes)
    ps = random.choice(pitch_seps)
    tc_str = f"-{tc}" if tc else ""
    suffix = random.choice(["", " JASO", " (TYP)"])

    # FIX: quantity multiplier now also uses × (not just X) to match real labels
    n = random.randint(2, 8)
    qty_forms = ["", f"{n}X ", f"{n}× ", f"{n}x "]
    qty = random.choice(qty_forms)
    # FIX: pitch separator after qty can also be × giving: 4×M8×1.25
    if qty and "×" in qty:
        ps = random.choice(["×", "x"])  # keep consistent with prefix
    return f"{qty}M{d}{ps}{p}{tc_str}{suffix}"

# ── + (unilateral)  Unilateral positive tolerance ─────────────────────────────
def gen_unilateral_pos() -> str:
    # real: Ø10.05+0.05 mm
    val   = _val(1, 100, 2)
    symbol = random.choice(["+","-"])
    upper = round(random.uniform(0.01, 0.5), 2)
    space  = random.choice(["", " "])
    return f"{symbol}{space}{upper}".strip()

# ── ×  Dimension separator ────────────────────────────────────────────────────
def gen_dimension_pair() -> str:
    # real: 80 × 60 mm, 4X M8 x1.25
    a = _val(5, 500, random.choice([0, 1]))
    b = _val(5, 500, random.choice([0, 1]))
    sep  = random.choice(["×","X"])
    unit = random.choice(["mm", "mm", ""])
    return f"{a} {sep} {b} {unit}".strip()

# ── *  Footnote marker ────────────────────────────────────────────────────────
def gen_asterisk_note() -> str:
    # real: Dual SR 13 & 9 *
    val = _val(1, 50, random.choice([0, 1]))
    return random.choice([
        f"{val} *",
        f"{val}*",
    ])

# ── MAX / MIN modifiers ───────────────────────────────────────────────────────
def gen_max() -> str:
    # real: 11.0 MAX, 71 MAX, 184.0 MAX
    val  = _val(0.1, 500, random.choice([0, 1]))
    unit = random.choice(["", "mm"])
    return f"{val} {unit} MAX".strip() if unit else f"{val} MAX"

def gen_min() -> str:
    val  = _val(0.1, 500, random.choice([0, 1]))
    unit = random.choice(["", "mm", "mm"])
    return f"{val} {unit} MIN".strip() if unit else f"{val} MIN"

# ── REF  Reference dimension ──────────────────────────────────────────────────
def gen_ref() -> str:
    # real: 10.1 REF
    val = _val(0.1, 200, random.choice([0, 1, 2]))
    return f"{val} REF"

# ── Limit tolerance (stacked, rendered as slash) ──────────────────────────────
def gen_limit_tolerance() -> str:
    # real: 10.1/9.9, 30.2/29.8, 121.0/120.0
    nominal = _val(1, 200, random.choice([1, 2]))
    spread  = round(random.uniform(0.1, 2.0), 1)
    upper   = round(nominal + spread / 2, 1)
    lower   = round(nominal - spread / 2, 1)
    return f"{upper}/{lower}"

# ── TYP  Typical ──────────────────────────────────────────────────────────────
def gen_typ() -> str:
    # real: M10X1-6H (TYP)
    # generate a thread then append (TYP)
    base = gen_thread().replace(" (TYP)", "")
    return f"{base} (TYP)"

# ── nX  Quantity multiplier (standalone) ─────────────────────────────────────
def gen_qty_multiplier() -> str:
    # real: 4X M8 x1.25
    n    = random.randint(2, 12)
    base = gen_thread()
    return f"{n}X {base}"

# ── Plain linear dimension ────────────────────────────────────────────────────
def gen_linear() -> str:
    val  = _val(0.5, 500, random.choice([0, 1, 2]))
    tol  = _tol()
    unit = random.choice(["mm", ""])
    forms = [
        f"{val}",
        f"{val} {unit}".strip(),
        f"{val}±{tol}",
        f"{val}±{tol} {unit}".strip(),
        f"{val}+{tol}/-0",
    ]
    return random.choice(forms)

# ── =  Equality / tolerance class (ensures = appears in training) ─────────────
def gen_equals() -> str:
    # = appears in tolerance classes (6H), fit specs, and equivalence notes
    val = _val(0.5, 200, random.choice([0, 1, 2]))
    return random.choice([
        f">={val}",               # >=1.67
        f"<={val}",               # <=0.05
        f"={val}mm",              # =10.5mm (reference callout)
        f"={val}",
        f"+{val}",
        f"-{val}",
    ])

# ── -  Hyphen / minus (ensures - appears in training) ────────────────────────
def gen_hyphen() -> str:
    # - appears in thread tolerance classes, range specs, part numbers
    val1 = _val(1, 100, random.choice([0, 1]))
    val2 = _val(1, 100, random.choice([0, 1]))
    tol_class = random.choice(["6H", "6g", "6f", "7H", "5H"])
    d = random.choice([6, 8, 10, 12, 16, 20])
    p = random.choice([0.75, 1.0, 1.25, 1.5])
    return random.choice([
        f"M{d}x{p}-{tol_class}",  # M12x1-6H
        f"{val1}-{val2}mm",        # 10-12mm (range)
        f"{val1}-{val2}",          # 10-12
    ])

# ── +  Plus / unilateral upper (ensures + appears standalone in training) ─────
def gen_plus() -> str:
    val   = _val(1, 200, random.choice([0, 1, 2]))
    upper = round(random.uniform(0.01, 1.0), 2)
    lower = round(random.uniform(0.01, 1.0), 2)
    return random.choice([
        f"{val}+{upper}/-0",       # 10.05+0.05/-0
        f"{val}+{upper}/-{lower}", # 10.05+0.05/-0.02
        f"+{upper}",               # isolated upper tol
        f"{val}+{upper}",          # 10+0.5
        f"+{val}",
    ])

# ── ( ) Parentheses (ensures model learns to read them around text) ─────────────
def gen_parentheses() -> str:
    val = _val(1, 100, random.choice([0, 1]))
    return random.choice([
        f"({val})",
        f"({val} REF)",
        f"({val} MAX)",
        "(TYP)",
        "(SEE NOTE)",
        "(2 PLACES)"
    ])


# ── Coordinate tuple  (x,y±tol) ──────────────────────────────────────────────
def gen_coordinate() -> str:
    # FIX: real labels like (67.0,76.0±5), (120.5,34.0±2)
    x   = _val(1, 300, random.choice([0, 1]))
    y   = _val(1, 300, random.choice([0, 1]))
    tol = _val(0.5, 10, random.choice([0, 1]))
    sep = random.choice(["", " "])
    return random.choice([
        f"({x},{y}±{tol})",
        f"({x},{sep}{y}±{tol})",
        f"({x},{y})",
    ])


# ── µ  Micro prefix (ensures µ appears, not just µm compound) ────────────────
def gen_micro() -> str:
    val = _val(0.1, 1000, random.choice([0, 1]))
    return random.choice([
        f"{val} µm",    # 500 µm
        f"{val}µm",     # 500µm
        f"<{val} µm",   # <500 µm
        f"±{val} µm"   # ±0.5 µm    # capacitance (rare but uses same char)
    ])

# ── Random Part Numbers (trains A-Z and 0-9) ──────────────────────────────────
def gen_part_number() -> str:
    prefix = random.choice(["PN ", "P/N ", "PART NO. ", ""])
    num = "".join(random.choices("0123456789", k=random.randint(4, 7)))
    suffix = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 3)))
    sep = random.choice(["-", "", " "])
    return f"{prefix}{num}{sep}{suffix}"

# ── Generic CAD Notes (trains full English words) ──────────────────────────────
def gen_cad_notes() -> str:
    notes = [
        "SEE DETAIL A",
        "ALL DIMS IN MM",
        "MATERIAL: STEEL",
        "SCALE 1:1",
        "SECTION B-B",
        "REMOVE BURRS",
        "TOLERANCE ±0.1",
        "QTY: 4",
        "SURFACE FINISH",
        "DO NOT SCALE"
    ]
    return random.choice(notes)


# ---------------------------------------------------------------------------
# Generator table — (fn, relative_weight)
# ---------------------------------------------------------------------------
GENERATORS = [
    (gen_diameter,        20),   # most common in CAD drawings (includes spaced Ø)
    (gen_plusminus,       3),    # bumped: ± confusion was a failure case
    (gen_linear,          2),
    (gen_angle,           8),    # bumped: now includes toleranced angle 35°±3°
    (gen_thread,          3),    # bumped: now includes × quantity prefix
    (gen_radius,          1),
    (gen_greater_than,    3),    # bumped: > + % combos
    (gen_less_than,       3),    # bumped: symmetric with greater_than
    (gen_limit_tolerance, 1),
    (gen_max,             1),
    (gen_min,             1),
    (gen_dimension_pair,  3),    # bumped: × separator coverage
    (gen_micrometre,      1),
    (gen_ref,             1),
    (gen_unilateral_pos,  1),
    (gen_equals,          1),    # ensures = in character set
    (gen_hyphen,          1),    # ensures - in character set
    (gen_plus,            1),    # ensures + in character set
    (gen_micro,           1),    # ensures µ in character set
    (gen_gte,             8),    # bumped: ≥ was a failure case, was missing from dict
    (gen_lte,             8),    # new: ≤ symmetric coverage
    (gen_percentage,      6),    # new: % was missing from dict
    (gen_inches,          3),
    (gen_sph_radius,      1),
    (gen_typ,             1),
    (gen_qty_multiplier,  2),    # bumped
    (gen_asterisk_note,   1),
    (gen_parentheses,     5),    # ensures () appears around random numbers and text
    (gen_coordinate,      5),    # FIX: (67.0,76.0±5) coordinate tuple format
    (gen_part_number,     10),   # high weight to train English characters heavily
    (gen_cad_notes,       10),   # high weight to train English words heavily
]

def sample_text() -> str:
    fns, weights = zip(*GENERATORS)
    return random.choices(fns, weights=weights, k=1)[0]()


# ---------------------------------------------------------------------------
# 2. Rendering
# ---------------------------------------------------------------------------

def load_fonts(font_dir: Path, sizes: list[int]) -> list[ImageFont.FreeTypeFont]:
    """Load all TTF/OTF fonts from a directory at multiple sizes."""
    fonts = []
    font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    if not font_files:
        print(f"[warn] No fonts found in {font_dir}, using PIL default.", file=sys.stderr)
    for fp in font_files:
        for sz in sizes:
            try:
                fonts.append(ImageFont.truetype(str(fp), sz))
            except Exception:
                pass
    if not fonts:
        fonts = [ImageFont.load_default()]
    return fonts


def render_text(text: str, font: ImageFont.FreeTypeFont, padding: int = 6) -> np.ndarray:
    """Render text to a tight-cropped grayscale numpy array (uint8)."""
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + padding * 2
    h = bbox[3] - bbox[1] + padding * 2
    if w < 4 or h < 4:
        return None

    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=0)
    return np.array(img)


# ---------------------------------------------------------------------------
# 3. Augmentations
# ---------------------------------------------------------------------------

def aug_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, random.uniform(2, 15), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def aug_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def aug_rotate(img: np.ndarray) -> np.ndarray:
    angle = random.uniform(-4, 4)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)

def aug_scale(img: np.ndarray) -> np.ndarray:
    scale = random.uniform(0.97, 1.25)
    h, w = img.shape
    nh, nw = max(4, int(h * scale)), max(4, int(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

def aug_brightness(img: np.ndarray) -> np.ndarray:
    delta = random.randint(-40, 40)
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

def aug_jpeg(img: np.ndarray) -> np.ndarray:
    q = random.randint(35, 80)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def aug_thin_lines(img: np.ndarray) -> np.ndarray:
    """Simulate dimension/leader lines bleeding into the text region."""
    out = img.copy()
    h, w = out.shape
    for _ in range(random.randint(1, 3)):
        if random.random() < 0.5:  # vertical
            x = random.randint(0, w - 1)
            cv2.line(out, (x, 0), (x, h - 1), random.randint(100, 200), 1)
        else:  # horizontal
            y = random.randint(0, h - 1)
            cv2.line(out, (0, y), (w - 1, y), random.randint(100, 200), 1)
    return out

def aug_perspective(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    d = random.randint(1, min(5, h // 4, w // 4))
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), random.randint(0, d)],
        [random.randint(0, d), h - random.randint(0, d)],
        [w - random.randint(0, d), h - random.randint(0, d)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255)

def aug_dilate_erode(img: np.ndarray) -> np.ndarray:
    """Thicken or thin strokes to simulate print variation."""
    k = np.ones((2, 2), np.uint8)
    if random.random() < 0.5:
        return cv2.dilate(img, k, iterations=1)
    else:
        return cv2.erode(img, k, iterations=1)

def aug_shadow(img: np.ndarray) -> np.ndarray:
    """Gradient shadow across part of the image."""
    h, w = img.shape
    shadow = np.linspace(0, random.randint(30, 80), w).astype(np.uint8)
    shadow = np.tile(shadow, (h, 1))
    if random.random() < 0.5:
        shadow = shadow.T[:h, :w]
    return np.clip(img.astype(np.int16) - shadow, 0, 255).astype(np.uint8)


AUGMENTATIONS = [
    (aug_noise,         4),
    (aug_blur,          3),
    (aug_rotate,        5),
    (aug_scale,         12),
    (aug_brightness,    3),
    (aug_jpeg,          4),
    (aug_thin_lines,    3),
    (aug_perspective,   3),
    (aug_dilate_erode,  2),
    (aug_shadow,        3)
]

def augment(img: np.ndarray, n: int = 3) -> np.ndarray:
    fns, weights = zip(*AUGMENTATIONS)
    chosen = random.choices(fns, weights=weights, k=n)
    for fn in chosen:
        try:
            img = fn(img)
        except Exception:
            pass  # skip bad augmentation silently
    return img


# ---------------------------------------------------------------------------
# 4. Background compositing
# ---------------------------------------------------------------------------

def load_bg_patches(bg_dir: Path) -> list[np.ndarray]:
    patches = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        for p in bg_dir.glob(ext):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                patches.append(img)
    return patches


def composite_on_bg(text_img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    h, w = text_img.shape
    bh, bw = bg.shape

    # Tile bg if too small
    if bh < h or bw < w:
        reps_y = (h // bh) + 2
        reps_x = (w // bw) + 2
        bg = np.tile(bg, (reps_y, reps_x))
        bh, bw = bg.shape

    y = random.randint(0, bh - h)
    x = random.randint(0, bw - w)
    patch = bg[y:y + h, x:x + w].astype(np.float32)

    # Multiply blend: dark ink on textured paper
    text_norm = text_img.astype(np.float32) / 255.0
    blended = (text_norm * (patch / 255.0) * 255.0).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# 5. Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    corpus: list[str],
    fonts: list[ImageFont.FreeTypeFont],
    out_dir: Path,
    bg_patches: list[np.ndarray],
    n_aug: int,
    bg_prob: float = 0.5,
) -> None:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    label_lines = []
    skipped = 0

    total = len(corpus) * n_aug
    for i, text in enumerate(corpus):
        font = random.choice(fonts)
        base = render_text(text, font)
        if base is None:
            skipped += 1
            continue

        for j in range(n_aug):
            aug = augment(base.copy(), n=random.randint(2, 4))

            if bg_patches and random.random() < bg_prob:
                bg = random.choice(bg_patches)
                aug = composite_on_bg(aug, bg)

            fname = f"syn_{i:06d}_{j}.png"
            cv2.imwrite(str(img_dir / fname), aug)
            label_lines.append(f"images/{fname}\t{text}")

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(corpus)} texts rendered ({(i+1)*n_aug} images)...")

    label_path = out_dir / "rec_gt.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))

    print(f"\nDone.")
    print(f"  Images : {len(label_lines)} → {img_dir}")
    print(f"  Labels : {label_path}")
    if skipped:
        print(f"  Skipped: {skipped} texts (font missing glyph?)")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Synthetic CAD OCR data generator")
    p.add_argument("--out_dir",     default="./synthetic_data",  help="Output directory")
    p.add_argument("--fonts",       default=None,                help="Directory of TTF/OTF fonts")
    p.add_argument("--real_bg_dir", default=None,                help="Directory of real CAD bg patches (optional)")
    p.add_argument("--n_texts",     type=int, default=5000,      help="Number of unique text strings to generate")
    p.add_argument("--n_aug",       type=int, default=5,         help="Augmented variants per text string")
    p.add_argument("--bg_prob",     type=float, default=0.5,     help="Probability of compositing on real bg")
    p.add_argument("--font_sizes",  default="24,32,40,48",       help="Comma-separated font sizes to use")
    p.add_argument("--seed",        type=int, default=42,        help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir    = Path(args.out_dir)
    font_sizes = [int(s) for s in args.font_sizes.split(",")]

    print(f"=== Synthetic CAD OCR Generator ===")
    print(f"Output : {out_dir}")
    print(f"Texts  : {args.n_texts}  ×  {args.n_aug} augmentations = {args.n_texts * args.n_aug} images")

    # Fonts
    if args.fonts:
        font_dir = Path(args.fonts)
        fonts = load_fonts(font_dir, font_sizes)
    else:
        fonts = [ImageFont.load_default()]
    print(f"Fonts  : {len(fonts)} loaded")

    # Background patches
    bg_patches = []
    if args.real_bg_dir:
        bg_patches = load_bg_patches(Path(args.real_bg_dir))
        print(f"BG patches: {len(bg_patches)} loaded")
    else:
        print("BG patches: none (pass --real_bg_dir to enable compositing)")

    # Generate corpus
    print(f"\nGenerating {args.n_texts} CAD strings...")
    corpus = [sample_text() for _ in range(args.n_texts)]

    # Preview sample
    print("Sample strings:", corpus[:8])

    # Build
    print(f"\nRendering + augmenting...")
    build_dataset(
        corpus    = corpus,
        fonts     = fonts,
        out_dir   = out_dir,
        bg_patches= bg_patches,
        n_aug     = args.n_aug,
        bg_prob   = args.bg_prob,
    )


if __name__ == "__main__":
    main()