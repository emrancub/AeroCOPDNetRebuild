from __future__ import annotations

# ---- path bootstrap: allow `from src...` when running as `python scripts\x.py`
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---- end bootstrap

import argparse, copy
from pathlib import Path
from src.utils.io import read_yaml, write_yaml
import subprocess, sys, yaml

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
    # inherit resolution handled by run_cv, but here we expand variants
    variants = cfg["ablation"]["variants"]
    for v in variants:
        vcfg = copy.deepcopy(cfg)
        name = v["name"]
        if "augment" in v:
            vcfg["augment"] = deep_merge(vcfg.get("augment", {}), v["augment"])
        vcfg["outputs"]["experiment_name"] = f"{cfg['outputs']['experiment_name']}_{name}"
        tmp = Path("configs") / f"_tmp_ablation_aug_{name}.yaml"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(vcfg, f, sort_keys=False)
        subprocess.check_call([sys.executable, "scripts/run_cv.py", "--config", str(tmp)])
        tmp.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
