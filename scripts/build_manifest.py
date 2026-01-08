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
from src.data.manifest_builder import (
    build_icbhi_binary, build_fraiwan_binary, pooled_manifest, diagnosis_breakdown
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icbhi_root", type=str, required=True)
    ap.add_argument("--fraiwan_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--icbhi_audio_subdir", type=str, default="audio_and_txt_files")
    ap.add_argument("--icbhi_diag_csv", type=str, default="patient_diagnosis.csv")

    ap.add_argument("--fraiwan_audio_subdir", type=str, default="Audio Files")
    ap.add_argument("--fraiwan_anno_xlsx", type=str, default="Data annotation.xlsx")
    ap.add_argument("--fraiwan_window_sec", type=float, default=5.0)
    ap.add_argument("--fraiwan_hop_sec", type=float, default=2.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    icbhi_csv = out_dir / "icbhi_binary.csv"
    fraiwan_csv = out_dir / "fraiwan_binary.csv"
    pooled_csv = out_dir / "pooled_binary.csv"

    build_icbhi_binary(Path(args.icbhi_root), icbhi_csv, args.icbhi_audio_subdir, args.icbhi_diag_csv)
    build_fraiwan_binary(Path(args.fraiwan_root), fraiwan_csv, args.fraiwan_audio_subdir, args.fraiwan_anno_xlsx,
                         args.fraiwan_window_sec, args.fraiwan_hop_sec)
    pooled_manifest(icbhi_csv, fraiwan_csv, pooled_csv)

    diagnosis_breakdown(icbhi_csv, out_dir / "breakdown_icbhi")
    diagnosis_breakdown(fraiwan_csv, out_dir / "breakdown_fraiwan")
    diagnosis_breakdown(pooled_csv, out_dir / "breakdown_pooled")

    print("Wrote:", icbhi_csv)
    print("Wrote:", fraiwan_csv)
    print("Wrote:", pooled_csv)
    print("Wrote breakdown tables under:", out_dir)

if __name__ == "__main__":
    main()
