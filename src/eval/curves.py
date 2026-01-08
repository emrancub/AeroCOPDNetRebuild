from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

def save_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr.tolist() + [np.nan] * (len(fpr) - len(thr))})
    roc_path = out_dir / "roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)

    prec, rec, thr2 = precision_recall_curve(y_true, y_prob)
    pr_df = pd.DataFrame({"precision": prec, "recall": rec, "threshold": thr2.tolist() + [np.nan] * (len(prec) - len(thr2))})
    pr_path = out_dir / "pr_curve.csv"
    pr_df.to_csv(pr_path, index=False)
    return roc_path, pr_path

def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0,1,n_bins+1)
    ids = np.digitize(y_prob, bins) - 1
    ids = np.clip(ids, 0, n_bins-1)
    rows=[]
    for b in range(n_bins):
        m = ids == b
        if m.sum() == 0:
            continue
        avg_p = float(np.mean(y_prob[m]))
        frac_pos = float(np.mean(y_true[m]))
        rows.append({"bin": b, "avg_prob": avg_p, "frac_pos": frac_pos, "count": int(m.sum())})
    return pd.DataFrame(rows)
