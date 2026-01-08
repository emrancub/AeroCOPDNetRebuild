from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import GlobalAvgPool, AttentiveStatsPool

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)

class DWSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x

@dataclass
class AeroCOPDNetCfg:
    use_se: bool = True
    pooling: Literal["gap","asp"] = "gap"
    base_channels: int = 32

class AeroCOPDNet(nn.Module):
    def __init__(self, cfg: AeroCOPDNetCfg, n_mels: int = 64):
        super().__init__()
        c = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSeparableConv(c,   c,   stride=1, use_se=cfg.use_se),
            DWSeparableConv(c,   2*c, stride=2, use_se=cfg.use_se),
            DWSeparableConv(2*c, 2*c, stride=1, use_se=cfg.use_se),
            DWSeparableConv(2*c, 4*c, stride=2, use_se=cfg.use_se),
            DWSeparableConv(4*c, 4*c, stride=1, use_se=cfg.use_se),
        )
        if cfg.pooling == "gap":
            self.pool = GlobalAvgPool()
            head_in = 4*c
        else:
            self.pool = AttentiveStatsPool(4*c)
            head_in = 8*c

        self.head = nn.Sequential(
            nn.Linear(head_in, head_in//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(head_in//2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,F,T]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)          # [B,C] or [B,2C]
        logits = self.head(x).squeeze(1)
        return logits
