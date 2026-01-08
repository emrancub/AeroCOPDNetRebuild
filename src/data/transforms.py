from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

@dataclass
class SpecAugmentCfg:
    freq_mask: int = 8
    time_mask: int = 24

def spec_augment(mel: torch.Tensor, cfg: SpecAugmentCfg) -> torch.Tensor:
    # mel: [1, F, T]
    x = mel.clone()
    _, F, T = x.shape
    # frequency mask
    if cfg.freq_mask and cfg.freq_mask > 0:
        f = int(torch.randint(low=0, high=cfg.freq_mask + 1, size=(1,)).item())
        f0 = int(torch.randint(low=0, high=max(1, F - f + 1), size=(1,)).item())
        x[:, f0:f0+f, :] = 0.0
    # time mask
    if cfg.time_mask and cfg.time_mask > 0:
        t = int(torch.randint(low=0, high=cfg.time_mask + 1, size=(1,)).item())
        t0 = int(torch.randint(low=0, high=max(1, T - t + 1), size=(1,)).item())
        x[:, :, t0:t0+t] = 0.0
    return x

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix
