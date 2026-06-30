import os
import cv2
import torch
import numpy as np
from blueprint_ocr import CustomCRNN, greedy_ctc_decoder, load_vocab

SYNTHETIC_PREFIXES = ("syn_",)

def is_actual_dataset(img_rel_path):
    filename = os.path.basename(img_rel_path)
    return not any(filename.startswith(p) for p in SYNTHETIC_PREFIXES)

def generate_actual_dataset_report():
    model_weights = "best_model.pth"
    test_gt      = r"d:\Cogentic\sign-detection\dataset_nived\dataset\test\rec_gt.txt"
    test_dir     = r"d:\Cogentic\sign-detection\dataset_nived\dataset\test"
    output_html  = "actual_dataset_report.html"

    # ── Vocab & Model ─────────────────────────────────────────────────────────
    vocab_chars = load_vocab()
    idx2char    = {idx: char for idx, char in enumerate(vocab_chars)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CustomCRNN(num_classes=len(vocab_chars)).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    results = []
    correct = 0
    total   = 0

    print("─── Running Evaluation on Actual Dataset Images ───")

    with open(test_gt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            img_rel_path = parts[0].strip()
            ground_truth = parts[1].strip()

            # ── Skip synthetic entries ────────────────────────────────────────
            if not is_actual_dataset(img_rel_path):
                continue

            img_path = os.path.join(test_dir, img_rel_path)
            if not os.path.exists(img_path):
                print(f"  [MISSING] {img_path}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [UNREADABLE] {img_path}")
                continue

            resized   = cv2.resize(img, (128, 32))
            norm_img  = resized.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits  = model(img_tensor)
                decoded = greedy_ctc_decoder(logits.cpu(), idx2char)[0]

            is_correct = (decoded == ground_truth)
            if is_correct:
                correct += 1
            total += 1

            abs_img_path = "file:///" + img_path.replace("\\", "/")
            results.append((abs_img_path, ground_truth, decoded, is_correct))

    # Sort: incorrect first for easier debugging
    results.sort(key=lambda x: x[3])

    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"  Total actual images: {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print("Generating HTML …")

    # ── HTML ──────────────────────────────────────────────────────────────────
    acc_color = "#27ae60" if accuracy >= 70 else "#e74c3c"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Actual Dataset – OCR Evaluation</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Inter', sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 32px 24px;
    }}

    /* ── Header ── */
    .header {{
      text-align: center;
      margin-bottom: 40px;
    }}
    .header h1 {{
      font-size: 2rem;
      font-weight: 700;
      background: linear-gradient(135deg, #a78bfa, #60a5fa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 6px;
    }}
    .header p {{
      color: #64748b;
      font-size: 0.9rem;
    }}

    /* ── Summary Cards ── */
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
      max-width: 900px;
      margin: 0 auto 40px;
    }}
    .card {{
      background: #1e2130;
      border: 1px solid #2d3148;
      border-radius: 14px;
      padding: 22px 20px;
      text-align: center;
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }}
    .card .value {{
      font-size: 2.2rem;
      font-weight: 700;
      line-height: 1;
      margin-bottom: 6px;
    }}
    .card .label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      font-weight: 500;
    }}
    .card.total   .value {{ color: #60a5fa; }}
    .card.correct .value {{ color: #34d399; }}
    .card.wrong   .value {{ color: #f87171; }}
    .card.acc     .value {{ color: {acc_color}; }}

    /* ── Filters ── */
    .controls {{
      max-width: 1100px;
      margin: 0 auto 20px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .filter-btn {{
      padding: 7px 18px;
      border-radius: 999px;
      border: 1px solid #2d3148;
      background: #1e2130;
      color: #94a3b8;
      cursor: pointer;
      font-size: 0.82rem;
      font-family: inherit;
      transition: all 0.18s;
    }}
    .filter-btn:hover, .filter-btn.active {{
      background: #6366f1;
      border-color: #6366f1;
      color: #fff;
    }}
    .search {{
      margin-left: auto;
      padding: 7px 14px;
      border-radius: 8px;
      border: 1px solid #2d3148;
      background: #1e2130;
      color: #e2e8f0;
      font-family: inherit;
      font-size: 0.82rem;
      outline: none;
      width: 220px;
      transition: border-color 0.18s;
    }}
    .search:focus {{ border-color: #6366f1; }}
    .search::placeholder {{ color: #475569; }}

    /* ── Table ── */
    .table-wrap {{
      max-width: 1100px;
      margin: 0 auto;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #2d3148;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}
    thead tr {{
      background: #1a1d2e;
    }}
    th {{
      padding: 13px 16px;
      text-align: left;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      font-weight: 600;
      border-bottom: 1px solid #2d3148;
    }}
    tbody tr {{
      background: #161827;
      transition: background 0.15s;
      border-bottom: 1px solid #1e2130;
    }}
    tbody tr:hover {{ background: #1e2130; }}
    tbody tr:last-child {{ border-bottom: none; }}
    td {{
      padding: 11px 16px;
      vertical-align: middle;
    }}

    /* image cell */
    td.img-cell img {{
      height: 44px;
      max-width: 200px;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid #2d3148;
      background: #fff;
      display: block;
    }}

    /* text cells */
    .mono {{
      font-family: 'Courier New', monospace;
      font-size: 0.82rem;
      color: #cbd5e1;
    }}

    /* badge */
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.04em;
    }}
    .badge.correct {{ background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }}
    .badge.incorrect {{ background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }}

    /* source tag */
    .source-tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 500;
      background: #1e2130;
      border: 1px solid #2d3148;
      color: #94a3b8;
    }}

    /* hidden rows */
    tr.hidden {{ display: none; }}

    /* footer */
    .footer {{
      text-align: center;
      margin-top: 48px;
      color: #334155;
      font-size: 0.75rem;
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>Actual Dataset — OCR Evaluation</h1>
  <p>75 real-world blueprint crop images · model: <code>best_model.pth</code></p>
</div>

<div class="summary">
  <div class="card total">
    <div class="value">{total}</div>
    <div class="label">Total Images</div>
  </div>
  <div class="card correct">
    <div class="value">{correct}</div>
    <div class="label">Correct</div>
  </div>
  <div class="card wrong">
    <div class="value">{total - correct}</div>
    <div class="label">Incorrect</div>
  </div>
  <div class="card acc">
    <div class="value">{accuracy:.1f}%</div>
    <div class="label">Accuracy</div>
  </div>
</div>

<div class="controls">
  <button class="filter-btn active" onclick="filterRows('all', this)">All</button>
  <button class="filter-btn" onclick="filterRows('incorrect', this)">Incorrect only</button>
  <button class="filter-btn" onclick="filterRows('correct', this)">Correct only</button>
  <input class="search" type="text" id="searchBox" placeholder="Search ground truth or prediction…" oninput="applySearch()" />
</div>

<div class="table-wrap">
  <table id="resultTable">
    <thead>
      <tr>
        <th>#</th>
        <th>Crop Image</th>
        <th>Filename</th>
        <th>Ground Truth</th>
        <th>Prediction</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
"""

    for i, (img_path, target, pred, is_correct) in enumerate(results, 1):
        status_class = "correct" if is_correct else "incorrect"
        status_icon  = "✓" if is_correct else "✗"
        status_text  = "Correct" if is_correct else "Incorrect"
        filename     = img_path.split("/")[-1]
        # detect source prefix for the tag
        prefix = filename.split("_crop_")[0] if "_crop_" in filename else filename.split("_")[0]

        html += f"""      <tr data-status="{status_class}" data-gt="{target.lower()}" data-pred="{pred.lower()}">
        <td style="color:#475569;font-size:0.78rem;">{i}</td>
        <td class="img-cell"><img src="{img_path}" alt="{filename}" /></td>
        <td><span class="source-tag">{prefix}</span>&nbsp;<span style="color:#475569;font-size:0.75rem;">{filename}</span></td>
        <td class="mono">{target}</td>
        <td class="mono">{pred}</td>
        <td><span class="badge {status_class}">{status_icon} {status_text}</span></td>
      </tr>
"""

    html += """    </tbody>
  </table>
</div>

<div class="footer">Generated by generate_actual_dataset_report.py · blueprint_ocr CRNN model</div>

<script>
  let currentFilter = 'all';

  function filterRows(filter, btn) {
    currentFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applySearch();
  }

  function applySearch() {
    const q = document.getElementById('searchBox').value.toLowerCase();
    document.querySelectorAll('#resultTable tbody tr').forEach(row => {
      const status = row.dataset.status;
      const gt     = row.dataset.gt   || '';
      const pred   = row.dataset.pred || '';
      const matchFilter = (currentFilter === 'all') || (status === currentFilter);
      const matchSearch = !q || gt.includes(q) || pred.includes(q);
      row.classList.toggle('hidden', !(matchFilter && matchSearch));
    });
  }
</script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Done! Report saved to '{output_html}'.")


if __name__ == "__main__":
    generate_actual_dataset_report()
