from __future__ import annotations
import torch
import torch.nn as nn

class GlobalAvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,F,T] -> [B,C]
        return x.mean(dim=(2,3))

class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling over time with mean+std.
    We first average over frequency to get [B,C,T], then compute attention over T.
    """
    def __init__(self, channels: int, attention_hidden: int = 128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(channels, attention_hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_hidden, channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,F,T]
        x = x.mean(dim=2)  # [B,C,T]
        a = self.att(x)    # [B,C,T]
        w = torch.softmax(a, dim=-1)  # [B,C,T]
        mean = torch.sum(w * x, dim=-1)  # [B,C]
        var  = torch.sum(w * (x - mean.unsqueeze(-1))**2, dim=-1)
        std  = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=1)  # [B,2C]
