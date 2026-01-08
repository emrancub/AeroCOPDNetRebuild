from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ..train.metrics import compute_binary_metrics
from .curves import save_roc_pr, reliability_curve
from ..utils.io import write_json

def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    out_dir: Path,
    fold: int,
    split_name: str,
) -> Dict[str, Any]:
    model.eval()
    probs=[]
    true=[]
    rows=[]
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            logits = model(x).detach().cpu().numpy()
            p = 1.0/(1.0 + np.exp(-logits))
            probs.append(p)
            true.append(y)
            for i in range(len(y)):
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "subject_id": batch["subject_id"][i],
                    "dataset": batch["dataset"][i],
                    "filepath": batch["filepath"][i],
                    "y_true": float(y[i]),
                    "y_prob": float(p[i]),
                    "fold": int(fold),
                    "split": str(split_name),
                })
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(true).astype(int)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    y_pred = (y_prob >= threshold).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(y_pred=y_pred).to_csv(out_dir / "predictions.csv", index=False)

    save_roc_pr(y_true, y_prob, out_dir)
    rel = reliability_curve(y_true, y_prob, n_bins=10)
    rel.to_csv(out_dir / "reliability_curve.csv", index=False)

    write_json(out_dir / "metrics_test.json", metrics)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics_test.csv", index=False)
    return metrics
