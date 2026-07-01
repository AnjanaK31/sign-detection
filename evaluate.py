"""
evaluate.py — Evaluate a trained CRNN model on a test set.

Usage:
    # Console output only
    python evaluate.py --data data/dataset/test --model best_model.pth

    # Generate HTML visual report
    python evaluate.py --data data/dataset/test --model best_model.pth --html report.html
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from cad_ocr import CRNN, load_vocab, greedy_ctc_decode


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CRNN on a test set")
    p.add_argument("--data",  required=True, help="Path to test split directory (contains rec_gt.txt and images)")
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--html",  default=None,  help="Optional: output path for HTML visual report")
    p.add_argument("--dict",  default=None,  help="Override path to dict.txt")
    return p.parse_args()


def load_model(model_path: str, device: torch.device, dict_path=None):
    vocab     = load_vocab(dict_path)
    idx2char  = {i: ch for i, ch in enumerate(vocab)}

    ckpt = torch.load(model_path, map_location=device)

    # Support both legacy (state_dict only) and new (full checkpoint) formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict  = ckpt["model_state_dict"]
        num_classes = ckpt.get("vocab_size", state_dict["classifier.weight"].shape[0])
    else:
        state_dict  = ckpt
        num_classes = state_dict["classifier.weight"].shape[0]

    # Trim idx2char if checkpoint vocab is smaller than current dict
    idx2char = {k: v for k, v in idx2char.items() if k < num_classes}

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model: {model_path}  |  vocab_size={num_classes}")
    return model, idx2char


def preprocess(img_path: Path) -> torch.Tensor:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 32))
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0).unsqueeze(0)   # [1, 1, 32, 128]


def evaluate(args):
    test_dir = Path(args.data)
    gt_file  = test_dir / "rec_gt.txt"
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, idx2char = load_model(args.model, device, args.dict)

    results  = []
    correct  = 0
    total    = 0

    print("Running evaluation...")
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            img_rel, gt_text = parts[0].strip(), parts[1].strip()
            img_path = test_dir / img_rel

            tensor = preprocess(img_path)
            if tensor is None:
                continue

            with torch.no_grad():
                logits  = model(tensor.to(device))
                decoded = greedy_ctc_decode(logits.cpu(), idx2char)[0]

            is_correct = decoded == gt_text
            if is_correct:
                correct += 1
            total += 1

            abs_path = "file:///" + str(img_path).replace("\\", "/")
            results.append((abs_path, gt_text, decoded, is_correct))

    accuracy = (correct / total * 100) if total else 0.0
    print(f"\n{'='*50}")
    print(f"  Total : {total}")
    print(f"  Correct : {correct}")
    print(f"  Incorrect : {total - correct}")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"{'='*50}\n")

    if args.html:
        _write_html(results, correct, total, accuracy, args.html)
        print(f"HTML report saved -> {args.html}")


def _write_html(results, correct, total, accuracy, out_path):
    results_sorted = sorted(results, key=lambda x: x[3])   # incorrect first
    acc_color = "#27ae60" if accuracy > 70 else "#e74c3c"

    rows = ""
    for img_path, gt, pred, ok in results_sorted:
        cls  = "correct" if ok else "incorrect"
        label = "Correct" if ok else "Incorrect"
        rows += (
            f"<tr>"
            f"<td><img src='{img_path}' alt='crop'/></td>"
            f"<td>{gt}</td><td>{pred}</td>"
            f"<td class='{cls}'>{label}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CAD OCR Evaluation Report</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f4f9;color:#333;margin:20px}}
h1{{text-align:center;color:#2c3e50}}
.summary{{display:flex;justify-content:space-around;background:#fff;padding:20px;
          border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.1);margin-bottom:20px}}
.box{{text-align:center}} .box h2{{margin:0;font-size:2em}} .box p{{margin:5px 0 0;color:#7f8c8d}}
table{{width:100%;border-collapse:collapse;background:#fff;
       box-shadow:0 4px 6px rgba(0,0,0,.1);border-radius:8px;overflow:hidden}}
th,td{{padding:15px;text-align:left;border-bottom:1px solid #ddd}}
th{{background:#2c3e50;color:#fff}} tr:hover{{background:#f1f1f1}}
.correct{{color:#27ae60;font-weight:bold}} .incorrect{{color:#e74c3c;font-weight:bold}}
img{{height:40px;border:1px solid #ccc}}
</style>
</head>
<body>
<h1>CAD OCR Evaluation Report</h1>
<div class="summary">
  <div class="box"><h2>{total}</h2><p>Total</p></div>
  <div class="box"><h2>{correct}</h2><p>Correct</p></div>
  <div class="box"><h2>{total-correct}</h2><p>Incorrect</p></div>
  <div class="box"><h2 style="color:{acc_color}">{accuracy:.2f}%</h2><p>Accuracy</p></div>
</div>
<table>
<tr><th>Image</th><th>Ground Truth</th><th>Prediction</th><th>Status</th></tr>
{rows}
</table>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    evaluate(parse_args())
