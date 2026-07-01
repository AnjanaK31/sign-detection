# CAD Text OCR — CRNN + CTC

A production-ready OCR pipeline for recognising text in **engineering/CAD blueprint crop images**.
Built on a CRNN (CNN + BiLSTM) architecture trained with CTC loss, tailored for the
character set found in technical drawings: tolerances, diameters, thread specifications,
and engineering symbols.

---

## Results

| Model | Test Set | Accuracy |
|---|---|---|
| Baseline | Original dataset | 96% |
| Fine-tuned | Expanded dataset (+3 symbols, +alt fonts) | **93.61%** |
| Baseline on expanded dataset | — | 76% |

> Fine-tuning improved performance by **+17.6 pp** on the harder expanded test set.

---

## Architecture

```
Input [B, 1, 32, 128]
      |
CNN Backbone
  5 blocks: Conv2d -> BatchNorm -> ReLU -> MaxPool
      | output: [B, 512, 1, 32]
      |
Reshape to sequence: [32, B, 512]
      |
BiLSTM (2 layers, hidden=256, bidirectional)
      | output: [32, B, 512]
      |
Linear classifier -> [32, B, num_classes]
      |
CTC decode -> text string
```

- **Loss:** CTC (zero_infinity=True)
- **Optimizer:** Adam
- **Input size:** 32 x 128 grayscale
- **Vocab size:** 90 (89 characters + CTC blank)

---

## Vocabulary

Defined in `synthetic_generator/dict.txt`. 89 characters covering:

| Category | Characters |
|---|---|
| Engineering symbols | `± Ø ° × µ ≥ ≤ ⊕ ☒ ☐ ⌖` |
| Operators / punctuation | `> < = + - * / % ( ) , . : & "` |
| Alphanumeric | `A–Z  a–z  0–9` |
| Space | ` ` |

---

## Repository Structure

```
cad-ocr/
|
+-- cad_ocr/                      Python package (core library)
|   +-- __init__.py               Public API exports
|   +-- model.py                  CRNN architecture
|   +-- dataset.py                CADTextDataset, collate_fn, load_vocab
|   +-- decoder.py                greedy_ctc_decode
|
+-- train.py                      Training entry point (argparse)
+-- evaluate.py                   Evaluation + optional HTML report
+-- infer.py                      Single-image inference
|
+-- tools/
|   +-- split_dataset.py          80/20 train/test split
|   +-- merge_synthetic.py        Merge synthetic into real dataset
|   +-- purge_synthetic.py        Remove synthetic entries from splits
|
+-- synthetic_generator/
|   +-- gen_cad.py                Main synthetic data generator
|   +-- gen_altfont.py            Alt-font test image generator
|   +-- split.py                  Split synthetic data 80/20
|   +-- update_dict.py            Update vocab from GT files
|   +-- dict.txt                  Master character vocabulary
|   +-- fonts/                    Font files (gitignored)
|
+-- data/                         Dataset root (gitignored - not distributed)
+-- best_model.pth                Current best weights (Git LFS)
+-- best_model_backup_v1.pth      Pre-expansion backup (Git LFS)
+-- .gitignore
+-- README.md
```

---

## Setup

### Requirements

```bash
pip install torch torchvision opencv-python pillow numpy
```

### Clone (includes model weights via Git LFS)

```bash
git clone https://github.com/AnjanaK31/sign-detection.git
cd sign-detection
git lfs pull          # downloads best_model.pth (~20 MB)
```

---

## Usage

### Train

```bash
# From scratch
python train.py --data data/dataset --epochs 50 --batch 8 --lr 1e-4

# Fine-tune from a checkpoint
python train.py --data data/dataset --epochs 10 --lr 1e-5 --resume best_model.pth
```

The checkpoint saved by `train.py` stores full state (epoch, optimizer, loss)
so training can be resumed at any point.

### Evaluate

```bash
# Console summary only
python evaluate.py --data data/dataset/test --model best_model.pth

# With HTML visual report
python evaluate.py --data data/dataset/test --model best_model.pth --html report.html
```

### Inference (single image)

```bash
python infer.py --model best_model.pth --image path/to/crop.jpg
```

---

## Data Pipeline

```
1. Generate synthetic CAD text images
   python synthetic_generator/gen_cad.py --out_dir synthetic_generator/synthetic_data \
       --fonts synthetic_generator/fonts --n_texts 5000 --n_aug 5

2. Split synthetic data
   python synthetic_generator/split.py

3. Split real crop dataset
   python tools/split_dataset.py --data data/dataset

4. Merge synthetic into real
   python tools/merge_synthetic.py \
       --syn  synthetic_generator/synthetic_data \
       --real data/dataset

5. Train
   python train.py --data data/dataset --epochs 50

6. Evaluate
   python evaluate.py --data data/dataset/test --model best_model.pth --html report.html
```

---

## Dataset Composition

| Split | Real Crops | Synthetic | Total |
|---|---|---|---|
| Train | ~300 | ~32,000 | ~32,300 |
| Test | ~75 | ~8,600 | ~8,675 |

The synthetic generator (`gen_cad.py`) produces realistic CAD annotation strings
such as `Ø26.99mm`, `±0.05`, `M3x2.0-6f (TYP)`, `>=94.72mm`, `<163.0%`, `101.8 REF`
rendered across multiple fonts and augmentation settings.

The alt-font generator (`gen_altfont.py`) adds test images rendered in fonts not seen
during training (Space Mono, Roboto Mono), stress-testing generalisation.

---

## Notes

- Model weights are stored via **Git LFS** — run `git lfs pull` after cloning.
- Dataset images are **gitignored** and not distributed with this repo.
- The checkpoint format saves `epoch`, `model_state_dict`, `optimizer_state_dict`,
  and `vocab_size` for full resumability.
