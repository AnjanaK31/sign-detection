"""
cad_ocr/decoder.py
Greedy CTC decoder for sequence-to-text conversion.
"""

import torch


def greedy_ctc_decode(logits: torch.Tensor, idx2char: dict, blank_idx: int = 0) -> list[str]:
    """
    Greedy (best-path) CTC decoder.

    Args:
        logits:    Tensor of shape [seq_len, batch, num_classes]
        idx2char:  Mapping from class index to character string
        blank_idx: Index of the CTC blank token (default 0)

    Returns:
        List of decoded strings, one per batch item.
    """
    # [seq_len, batch] -> [batch, seq_len]
    preds = torch.argmax(logits, dim=2).permute(1, 0)

    results = []
    for seq in preds:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != blank_idx and idx != prev:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        results.append("".join(chars))

    return results
