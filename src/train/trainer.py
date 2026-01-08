from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import make_bce_loss
from .metrics import compute_binary_metrics
from ..data.transforms import spec_augment, SpecAugmentCfg, mixup_batch

@dataclass
class TrainResult:
    best_ckpt: Path
    curves: List[Dict[str, Any]]

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: Optional[float],
    scheduler_cfg: Dict[str, Any],
    amp: bool,
    aug_cfg: Dict[str, Any],
    threshold: float,
    save_dir: Path,
) -> TrainResult:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = make_bce_loss(pos_weight)
    loss_fn = loss_fn.to(device)

    sched = None
    if scheduler_cfg.get("scheduler","plateau") == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(scheduler_cfg.get("plateau_factor",0.5)),
            patience=int(scheduler_cfg.get("plateau_patience",5)),
            verbose=False,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type=="cuda")

    use_specaug = bool(aug_cfg.get("use_specaug", False))
    use_mixup   = bool(aug_cfg.get("use_mixup", False))
    mix_alpha   = float(aug_cfg.get("mixup_alpha", 0.2))
    mix_prob    = float(aug_cfg.get("mixup_prob", 1.0))
    specaug = SpecAugmentCfg(
        freq_mask=int(aug_cfg.get("specaug_freq_mask",8)),
        time_mask=int(aug_cfg.get("specaug_time_mask",24)),
    )

    best_val = float("inf")
    best_ckpt = save_dir / "best.pt"
    curves: List[Dict[str, Any]] = []

    model.to(device)

    for epoch in range(1, epochs+1):
        model.train()
        tr_losses=[]
        tr_probs=[]
        tr_true=[]
        for batch in tqdm(train_loader, desc=f"Train ep{epoch}", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            if use_specaug:
                # apply per-sample specaug
                x = torch.stack([spec_augment(xi, specaug) for xi in x], dim=0)

            if use_mixup and np.random.rand() < mix_prob:
                x, y = mixup_batch(x, y, alpha=mix_alpha)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type=="cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_losses.append(loss.item())
            tr_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            tr_true.append(y.detach().cpu().numpy())

        tr_loss = float(np.mean(tr_losses))
        tr_probs = np.concatenate(tr_probs)
        tr_true  = np.concatenate(tr_true)
        tr_metrics = compute_binary_metrics(tr_true, tr_probs, threshold=threshold)

        # val
        model.eval()
        va_losses=[]
        va_probs=[]
        va_true=[]
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val ep{epoch}", leave=False):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                va_losses.append(loss.item())
                va_probs.append(torch.sigmoid(logits).cpu().numpy())
                va_true.append(y.cpu().numpy())
        va_loss=float(np.mean(va_losses))
        va_probs=np.concatenate(va_probs)
        va_true=np.concatenate(va_true)
        va_metrics=compute_binary_metrics(va_true, va_probs, threshold=threshold)

        if sched is not None:
            sched.step(va_loss)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_accuracy": tr_metrics["accuracy"],
            "val_accuracy": va_metrics["accuracy"],
            "train_auroc": tr_metrics["auroc"],
            "val_auroc": va_metrics["auroc"],
        }
        curves.append(row)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va_loss}, best_ckpt)

    # final ckpt
    torch.save({"model": model.state_dict(), "epoch": epochs, "val_loss": best_val}, save_dir / "final.pt")
    return TrainResult(best_ckpt=best_ckpt, curves=curves)
