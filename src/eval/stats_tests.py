from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def paired_tests(metric_csv_a: Path, metric_csv_b: Path, metric: str, out_csv: Path) -> None:
    a = pd.read_csv(metric_csv_a)
    b = pd.read_csv(metric_csv_b)
    # expect one row per fold, same ordering
    x = a[metric].values
    y = b[metric].values
    t_p = ttest_rel(x, y, nan_policy="omit").pvalue
    try:
        w_p = wilcoxon(x, y).pvalue
    except Exception:
        w_p = float("nan")
    out = pd.DataFrame([{"metric": metric, "paired_t_pvalue": t_p, "wilcoxon_pvalue": w_p}])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
