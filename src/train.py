"""
Train the LSTM autoencoder on normal TEP windows.

Outputs (under src/../models/):
  best_model.pt        weights of best-val-loss epoch
  training_log.csv     per-epoch train/val MSE
  training_curve.png   plot of train + val loss
  training.log         human-readable run log
"""

from __future__ import annotations
from pathlib import Path
import sys
import time
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from model import LSTMAutoencoder, count_params

# ---- Config ----
SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 60
PATIENCE = 8
HIDDEN_DIM = 64
N_LAYERS = 2
DROPOUT = 0.0
USE_AMP = True

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str, file_handle) -> None:
    print(msg)
    file_handle.write(msg + "\n")
    file_handle.flush()


def main() -> None:
    set_seeds(SEED)

    log_path = MODELS / "training.log"
    f_log = open(log_path, "w", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[setup] device: {device}", f_log)
    if device.type == "cuda":
        log(f"[setup] gpu: {torch.cuda.get_device_name(0)}", f_log)

    # ---- Data ----
    log("[setup] loading arrays ...", f_log)
    X_train = np.load(PROC / "X_train.npy")
    X_val = np.load(PROC / "X_val.npy")
    log(f"[setup] X_train: {X_train.shape}, X_val: {X_val.shape}", f_log)

    train_ds = TensorDataset(torch.from_numpy(X_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float())

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = LSTMAutoencoder(
        n_features=52,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        window_size=50,
        dropout=DROPOUT,
    ).to(device)
    log(f"[setup] model parameters: {count_params(model):,}", f_log)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    config = {
        "seed": SEED, "batch_size": BATCH_SIZE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS, "dropout": DROPOUT, "amp": USE_AMP,
    }
    log(f"[setup] config: {json.dumps(config)}", f_log)

    # ---- Training loop ----
    history = []
    best_val = float("inf")
    epochs_since_improve = 0
    best_path = MODELS / "best_model.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for (xb,) in train_dl:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                xhat = model(xb)
                loss = criterion(xhat, xb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                    xhat = model(xb)
                    loss = criterion(xhat, xb)
                val_losses.append(loss.item())

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))
        dt = time.time() - t0

        improved = val_mse < best_val - 1e-6
        if improved:
            best_val = val_mse
            epochs_since_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "config": config, "val_mse": val_mse}, best_path)
            tag = " *"
        else:
            epochs_since_improve += 1
            tag = ""

        log(
            f"[epoch {epoch:02d}/{MAX_EPOCHS}] "
            f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
            f"time={dt:.1f}s{tag}",
            f_log,
        )

        history.append({"epoch": epoch, "train_mse": train_mse,
                        "val_mse": val_mse, "time_s": dt})

        if epochs_since_improve >= PATIENCE:
            log(f"[stop] no improvement for {PATIENCE} epochs, stopping.", f_log)
            break

    # ---- Save logs and curve ----
    log_csv = MODELS / "training_log.csv"
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_mse,val_mse,time_s\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_mse']:.6f},{h['val_mse']:.6f},{h['time_s']:.2f}\n")
    log(f"[save] {log_csv.name}", f_log)

    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [h["train_mse"] for h in history], label="train MSE", lw=1.5)
    plt.plot(epochs, [h["val_mse"] for h in history], label="val MSE", lw=1.5)
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled units)")
    plt.title("LSTM Autoencoder — training curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = MODELS / "training_curve.png"
    plt.savefig(curve_path, dpi=120)
    log(f"[save] {curve_path.name}", f_log)

    log(f"[done] best val MSE: {best_val:.6f}, weights: {best_path}", f_log)
    f_log.close()


if __name__ == "__main__":
    main()