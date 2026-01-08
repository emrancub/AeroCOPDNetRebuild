from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import torch

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def try_flops(model: torch.nn.Module, input_shape=(1,1,64,200)) -> float:
    # Return FLOPs (not MACs) if possible; otherwise NaN.
    x = torch.zeros(*input_shape)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x).total()
        return float(flops)
    except Exception:
        return float("nan")

def save_params_flops(model: torch.nn.Module, out_csv: Path, input_shape=(1,1,64,200)) -> None:
    p = count_params(model)
    f = try_flops(model, input_shape=input_shape)
    df = pd.DataFrame([{"params": int(p), "flops": float(f)}])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
