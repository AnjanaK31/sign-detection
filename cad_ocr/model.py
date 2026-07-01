"""
cad_ocr/model.py
CRNN model architecture for CAD text recognition.
"""

import torch.nn as nn


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for sequence recognition.

    Architecture:
        CNN backbone  -> feature maps [B, 512, 1, W]
        BiLSTM head   -> sequence features [W, B, 512]
        Linear        -> per-timestep class logits [W, B, num_classes]

    Input:  grayscale image tensor [B, 1, 32, 128]
    Output: logits tensor [32, B, num_classes]  (seq_len=32 after pooling)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ── CNN Backbone ────────────────────────────────────────────────────
        # Input: [B, 1, 32, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # -> [B, 64,  16, 64]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # -> [B, 128,  8, 32]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),         # -> [B, 256,  4, 32]

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),         # -> [B, 256,  2, 32]

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),         # -> [B, 512,  1, 32]
        )

        # ── Sequence head ────────────────────────────────────────────────────
        # BiLSTM: input=512, hidden=256, bidirectional -> output=512
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True)

        # Per-timestep classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 32, 128]
        Returns:
            logits: [seq_len, B, num_classes]
        """
        feat = self.cnn(x)                     # [B, 512, 1, W]
        b, c, h, w = feat.size()
        assert h == 1, f"Expected height 1 after CNN, got {h}"

        feat = feat.squeeze(2)                 # [B, 512, W]
        feat = feat.permute(2, 0, 1)           # [W, B, 512]

        out, _ = self.rnn(feat)                # [W, B, 512]
        logits = self.classifier(out)          # [W, B, num_classes]
        return logits
