"""
train.py — Training entry point for CAD text OCR.

Usage:
    # Train from scratch
    python train.py --data data/dataset --epochs 50

    # Resume / fine-tune from a checkpoint
    python train.py --data data/dataset --epochs 10 --resume best_model.pth --lr 1e-5

Speed optimisations enabled by default (when CUDA is available):
  - AMP (automatic mixed precision)  — ~2× GPU throughput
  - OneCycleLR scheduler             — faster convergence (fewer epochs needed)
  - torch.compile()                  — 10–30% GPU speedup (torch >= 2.0)
  - num_workers=4, pin_memory        — removes data-loading bottleneck
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cad_ocr import CRNN, CADTextDataset, collate_fn, greedy_ctc_decode


def parse_args():
    p = argparse.ArgumentParser(description="Train CRNN for CAD text recognition")
    p.add_argument("--data",       required=True,            help="Path to dataset root (contains train/ and test/)")
    p.add_argument("--epochs",     type=int, default=50,     help="Number of training epochs")
    p.add_argument("--batch",      type=int, default=32,     help="Batch size (default 32)")
    p.add_argument("--lr",         type=float, default=1e-4, help="Peak learning rate for OneCycleLR")
    p.add_argument("--save",       default="best_model.pth", help="Path to save best model")
    p.add_argument("--resume",     default=None,             help="Checkpoint to load weights from")
    p.add_argument("--dict",       default=None,             help="Override path to dict.txt")
    p.add_argument("--workers",    type=int, default=4,      help="DataLoader num_workers (default 4)")
    p.add_argument("--no-compile", action="store_true",      help="Disable torch.compile() even on torch>=2.0")
    return p.parse_args()


def train(args):
    data_root = Path(args.data)
    train_gt  = data_root / "train" / "rec_gt.txt"
    train_dir = data_root / "train"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"   # AMP only beneficial on GPU
    print(f"Device : {device}  |  AMP : {use_amp}")

    pin = device.type == "cuda"
    dataset    = CADTextDataset(train_gt, train_dir, dict_path=args.dict)
    dataloader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )

    model     = CRNN(num_classes=dataset.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # OneCycleLR — ramps up then cosine-anneals; converges faster than flat LR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs,
        pct_start=0.1,          # warm-up for 10% of training
        anneal_strategy="cos",
    )

    # AMP scaler (no-op on CPU)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # torch.compile — significant speedup on torch>=2.0 + CUDA
    compiled = False
    if not getattr(args, 'no_compile', False) and device.type == "cuda":
        try:
            major = int(torch.__version__.split(".")[0])
            if major >= 2:
                model = torch.compile(model)
                compiled = True
                print("torch.compile() enabled.")
        except Exception as e:
            print(f"torch.compile() skipped: {e}")

    start_epoch = 0
    best_loss   = float("inf")

    # ── Resume / fine-tune ─────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_loss   = ckpt.get("best_loss", float("inf"))
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
        else:
            # Legacy weights-only checkpoint
            model.load_state_dict(ckpt)
            print(f"Loaded weights from '{args.resume}' (no epoch info).")
    else:
        print("Training from scratch.")

    # ── Training loop ──────────────────────────────────────────────────────
    total_steps     = args.epochs * len(dataloader)
    completed_steps = 0
    train_start     = time.time()

    model.train()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_loss  = 0.0
        epoch_start = time.time()

        for i, (images, targets, target_lengths) in enumerate(dataloader):
            images         = images.to(device, non_blocking=pin)
            targets        = targets.to(device, non_blocking=pin)
            target_lengths = target_lengths.to(device, non_blocking=pin)

            optimizer.zero_grad()

            # Forward pass — AMP autocasts to fp16 on GPU, fp32 on CPU
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                seq_len      = logits.size(0)
                batch_sz     = logits.size(1)
                input_lengths = torch.full((batch_sz,), seq_len, dtype=torch.long, device=device)
                log_probs = nn.functional.log_softmax(logits, dim=2)
                loss      = criterion(log_probs, targets, input_lengths, target_lengths)

            # Backward + step through scaler (no-op when AMP disabled)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss      += loss.item()
            completed_steps += 1

            # Print every 50 steps (reduced from 10 — less overhead)
            if i % 50 == 0:
                elapsed       = time.time() - train_start
                avg_step      = elapsed / completed_steps
                eta_secs      = avg_step * (total_steps - completed_steps)
                eta_h, rem    = divmod(int(eta_secs), 3600)
                eta_m, eta_s  = divmod(rem, 60)
                el_h,  el_r   = divmod(int(elapsed), 3600)
                el_m,  el_s   = divmod(el_r, 60)
                cur_lr        = scheduler.get_last_lr()[0]

                print(
                    f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} | LR: {cur_lr:.2e} | "
                    f"Elapsed: {el_h:02d}:{el_m:02d}:{el_s:02d} | "
                    f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                )

                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        decoded = greedy_ctc_decode(model(images).cpu(), dataset.idx2char)
                    print("  Samples:", " | ".join(decoded[:2]))
                model.train()

        avg_loss   = epoch_loss / len(dataloader)
        epoch_secs = time.time() - epoch_start
        ep_m, ep_s = divmod(int(epoch_secs), 60)

        elapsed     = time.time() - train_start
        epochs_done = epoch - start_epoch + 1
        epochs_left = args.epochs - epochs_done
        eta_secs    = (elapsed / epochs_done) * epochs_left if epochs_done else 0
        eta_h, rem  = divmod(int(eta_secs), 3600)
        eta_m, eta_s = divmod(rem, 60)

        print(
            f"\n{'='*70}\n"
            f"  Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | "
            f"Time: {ep_m:02d}m {ep_s:02d}s | ETA: {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s"
            f"\n{'='*70}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":               epoch + 1,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":           best_loss,
                "vocab_size":          dataset.vocab_size,
            }, args.save)
            print(f"  Saved best model (loss {best_loss:.4f}) -> {args.save}")

    total = time.time() - train_start
    tot_h, rem = divmod(int(total), 3600)
    tot_m, tot_s = divmod(rem, 60)
    print(f"\nDone in {tot_h:02d}h {tot_m:02d}m {tot_s:02d}s. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train(parse_args())
