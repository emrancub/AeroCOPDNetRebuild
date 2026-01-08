from __future__ import annotations
import torch
from torch.optim import Adam

def make_optimizer(model, lr: float, weight_decay: float):
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
