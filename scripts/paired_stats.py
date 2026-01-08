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
from src.eval.stats_tests import paired_tests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_fold_metrics", type=str, required=True, help="CSV with one row per fold (e.g., paper_assets/exp__tag__fold_metrics.csv)")
    ap.add_argument("--b_fold_metrics", type=str, required=True)
    ap.add_argument("--metric", type=str, default="auroc")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()
    paired_tests(Path(args.a_fold_metrics), Path(args.b_fold_metrics), metric=args.metric, out_csv=Path(args.out_csv))
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
