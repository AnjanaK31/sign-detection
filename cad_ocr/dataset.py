"""
cad_ocr/dataset.py
Dataset, vocabulary, and DataLoader utilities for CAD text recognition.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_DEFAULT_DICT = Path(__file__).resolve().parent.parent / "synthetic_generator" / "dict.txt"


def load_vocab(dict_path: str | Path | None = None) -> list[str]:
    """
    Load character vocabulary from a dictionary file.

    The file should have one character per line.
    Index 0 is always reserved for the CTC blank token.

    Args:
        dict_path: Path to dict.txt. Defaults to synthetic_generator/dict.txt.

    Returns:
        List of characters where index 0 == 'blank'.
    """
    path = Path(dict_path) if dict_path else _DEFAULT_DICT
    vocab = ["blank"]

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.rstrip("\n")
                if ch and ch not in vocab:
                    vocab.append(ch)
    else:
        print(f"[WARNING] Vocab file not found: {path}. Using minimal fallback.")
        vocab += list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .-+")

    return vocab


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CADTextDataset(Dataset):
    """
    PyTorch Dataset for CAD text recognition.

    Expects a tab-delimited ground-truth file:
        <relative_image_path>\\t<label text>

    Images are read as grayscale, resized to `target_shape` (H x W),
    and normalised to [0, 1].

    Args:
        gt_filepath:  Path to rec_gt.txt
        base_dir:     Root directory that the relative image paths are resolved from.
        target_shape: (height, width) the model expects. Default (32, 128).
        dict_path:    Optional override for the vocabulary file.
    """

    def __init__(
        self,
        gt_filepath: str | Path,
        base_dir: str | Path,
        target_shape: tuple[int, int] = (32, 128),
        dict_path: str | Path | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.target_shape = target_shape

        vocab = load_vocab(dict_path)
        self.char2idx: dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char: dict[int, str] = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)

        self.samples: list[tuple[str, str]] = []
        with open(gt_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    self.samples.append((parts[0].strip(), parts[1].strip()))

    def _encode(self, text: str) -> list[int]:
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_rel, label = self.samples[idx]
        img_path = self.base_dir / img_rel

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.target_shape, dtype=np.uint8)

        h, w = self.target_shape
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)   # [1, H, W]

        encoded = self._encode(label)
        target = torch.tensor(encoded, dtype=torch.long)
        target_len = torch.tensor(len(encoded), dtype=torch.long)

        return img_tensor, target, target_len


def collate_fn(batch):
    """Collate variable-length label sequences for CTC loss."""
    images, targets, lengths = zip(*batch)
    return (
        torch.stack(images, 0),
        torch.cat(targets, 0),
        torch.stack(lengths, 0),
    )
