"""
cad_ocr — CAD Text OCR package.

Public API:
    CRNN         — model architecture
    CADTextDataset, collate_fn, load_vocab  — data pipeline
    greedy_ctc_decode  — inference decoder
"""

from .model import CRNN
from .dataset import CADTextDataset, collate_fn, load_vocab
from .decoder import greedy_ctc_decode

__all__ = [
    "CRNN",
    "CADTextDataset",
    "collate_fn",
    "load_vocab",
    "greedy_ctc_decode",
]
