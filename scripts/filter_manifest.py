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

def normalize_dx(dx: str) -> str:
    return str(dx).strip().lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--keep_dataset", type=str, default=None, help="ICBHI or FRAIWAN")
    ap.add_argument("--mode", type=str, required=True, choices=[
        "pooled_all",
        "copd_vs_healthy",
        "copd_vs_other_diseases",
        "custom_dx",
    ])
    ap.add_argument("--dx_list", type=str, nargs="*", default=None, help="For mode=custom_dx: keep these diagnoses (case-insensitive).")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    if args.keep_dataset:
        df = df[df["dataset"].astype(str).str.upper() == args.keep_dataset.upper()].copy()

    dx = df.get("diagnosis", pd.Series(["Unknown"]*len(df))).astype(str).map(normalize_dx)

    if args.mode == "pooled_all":
        out = df
    elif args.mode == "copd_vs_healthy":
        # keep COPD or Healthy only; relabel: COPD=1, Healthy=0
        keep = dx.isin(["copd","healthy","normal"])
        out = df[keep].copy()
        out["label"] = (dx[keep] == "copd").astype(int).values
    elif args.mode == "copd_vs_other_diseases":
        # exclude healthy/normal, keep COPD vs everything else
        keep = ~dx.isin(["healthy","normal"])
        out = df[keep].copy()
        out["label"] = (dx[keep] == "copd").astype(int).values
    else:
        if not args.dx_list:
            raise ValueError("mode=custom_dx requires --dx_list")
        keep_set = set([normalize_dx(x) for x in args.dx_list])
        out = df[dx.isin(keep_set)].copy()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv, "rows=", len(out), "subjects=", out["subject_id"].nunique())

if __name__ == "__main__":
    main()
