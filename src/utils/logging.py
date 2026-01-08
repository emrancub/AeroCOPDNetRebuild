from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from .io import write_json, write_yaml, save_df_csv

@dataclass
class RunPaths:
    run_dir: Path
    fold_dir: Path

def init_run_dir(root: Path, exp_name: str) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / exp_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def init_fold_dir(run_dir: Path, fold: int) -> Path:
    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir

def save_config(fold_dir: Path, config: Dict[str, Any]) -> None:
    write_yaml(fold_dir / "config_used.yaml", config)

def save_curves(fold_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    df = pd.DataFrame(rows)
    path = fold_dir / "curves_train_val.csv"
    save_df_csv(path, df)
    return path

def save_metrics(fold_dir: Path, metrics: Dict[str, Any]) -> None:
    df = pd.DataFrame([metrics])
    save_df_csv(fold_dir / "metrics_test.csv", df)
    write_json(fold_dir / "metrics_test.json", metrics)
