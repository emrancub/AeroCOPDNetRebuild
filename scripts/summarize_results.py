from __future__ import annotations

# ---- path bootstrap: allow `from src...` when running as `python scripts\x.py`
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---- end bootstrap

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

METRIC_FILE = "metrics_test.csv"

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        if c in ["fold"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = f"{df[c].mean():.4f} ± {df[c].std(ddof=1):.4f}"
    return pd.DataFrame([out])

def collect_metrics_from_run(run_dir: Path) -> list[pd.DataFrame]:
    """Return list of (tag, fold_df) pairs.
    - Standard case: run_dir/fold_*/metrics_test.csv
    - Baselines case: run_dir/fold_*/<model>/metrics_test.csv
    """
    out = []
    # standard
    standard_rows=[]
    for f in sorted(run_dir.glob("fold_*")):
        m = f / METRIC_FILE
        if m.exists():
            d = pd.read_csv(m).iloc[0].to_dict()
            d["fold"] = int(f.name.split("_")[1])
            standard_rows.append(d)
    if standard_rows:
        out.append(("main", pd.DataFrame(standard_rows)))

    # nested (baselines)
    # discover model folders under fold_0
    fold0 = sorted(run_dir.glob("fold_*"))
    if fold0:
        model_dirs = [p for p in (fold0[0]).iterdir() if p.is_dir()]
        for md in model_dirs:
            rows=[]
            for f in sorted(run_dir.glob("fold_*")):
                m = f / md.name / METRIC_FILE
                if m.exists():
                    d = pd.read_csv(m).iloc[0].to_dict()
                    d["fold"] = int(f.name.split("_")[1])
                    rows.append(d)
            if rows:
                out.append((md.name, pd.DataFrame(rows)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", type=str, default="outputs")
    ap.add_argument("--out_dir", type=str, default="paper_assets")
    args = ap.parse_args()

    outputs = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_summaries=[]
    for exp in sorted(outputs.glob("*")):
        if not exp.is_dir():
            continue
        runs = sorted(exp.glob("*"))
        if not runs:
            continue
        latest = runs[-1]

        tagged = collect_metrics_from_run(latest)
        for tag, df in tagged:
            if df.empty:
                continue
            df.to_csv(out_dir / f"{exp.name}__{tag}__fold_metrics.csv", index=False)
            s = summarize(df)
            s.insert(0, "experiment", exp.name)
            s.insert(1, "model_tag", tag)
            s.to_csv(out_dir / f"{exp.name}__{tag}__summary_mean_std.csv", index=False)
            exp_summaries.append(s)

    if exp_summaries:
        all_df = pd.concat(exp_summaries, ignore_index=True)
        all_df.to_csv(out_dir / "all_experiments_summary.csv", index=False)

    print("Wrote summaries to:", out_dir)

if __name__ == "__main__":
    main()
