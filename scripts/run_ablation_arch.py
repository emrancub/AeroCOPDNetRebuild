from __future__ import annotations

# ---- path bootstrap: allow `from src...` when running as `python scripts\x.py`
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---- end bootstrap

import argparse, copy, subprocess, sys, yaml
from pathlib import Path
from src.utils.io import read_yaml

def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k,v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = read_yaml(Path(args.config))
    variants = cfg["ablation"]["variants"]
    for v in variants:
        name = v["name"]
        vcfg = copy.deepcopy(cfg)
        if "train" in v:
            vcfg["train"] = deep_merge(vcfg.get("train", {}), v["train"])
        vcfg["outputs"]["experiment_name"] = f"{cfg['outputs']['experiment_name']}_{name}"
        tmp = Path("configs") / f"_tmp_ablation_arch_{name}.yaml"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(vcfg, f, sort_keys=False)
        subprocess.check_call([sys.executable, "scripts/run_cv.py", "--config", str(tmp)])
        tmp.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
