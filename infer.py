"""
infer.py — Run inference on a single crop image.

Usage:
    python infer.py --model best_model.pth --image path/to/crop.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from cad_ocr import CRNN, load_vocab, greedy_ctc_decode


def parse_args():
    p = argparse.ArgumentParser(description="CRNN inference on a single image")
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--image", required=True, help="Path to input crop image")
    p.add_argument("--dict",  default=None,  help="Override path to dict.txt")
    return p.parse_args()


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load vocab + model ─────────────────────────────────────────────────
    vocab    = load_vocab(args.dict)
    idx2char = {i: ch for i, ch in enumerate(vocab)}

    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict  = ckpt["model_state_dict"]
        num_classes = ckpt.get("vocab_size", state_dict["classifier.weight"].shape[0])
    else:
        state_dict  = ckpt
        num_classes = state_dict["classifier.weight"].shape[0]

    idx2char = {k: v for k, v in idx2char.items() if k < num_classes}

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Preprocess ─────────────────────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return

    img    = cv2.resize(img, (128, 32))
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, 32, 128]

    # ── Predict ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits  = model(tensor)
        decoded = greedy_ctc_decode(logits.cpu(), idx2char)[0]

    print(f"Prediction: '{decoded}'")


if __name__ == "__main__":
    infer(parse_args())
