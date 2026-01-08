from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

def make_bce_loss(pos_weight: Optional[float] = None) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    w = torch.tensor([pos_weight], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=w)
