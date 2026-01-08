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
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.io import read_yaml, write_json
from src.utils.repro import set_global_seed, get_device, environment_snapshot
from src.utils.logging import init_run_dir, init_fold_dir, save_config, save_curves, save_metrics
from src.data.features import FeatureConfig, compute_train_stats
from src.data.loaders_icbhi import LungSoundDataset
from src.train.trainer import train_one_fold
from src.eval.evaluate import evaluate_model
from src.eval.flops_params import save_params_flops
from src.eval.confusion import save_confusion
from src.utils.plotting import PlotStyle, plot_learning_curves

from scripts.run_cv import load_config, build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed, deterministic=True)
    device = get_device(cfg["project"]["device"])

    df = pd.read_csv(cfg["data"]["manifest_csv"])
    cd = cfg["cross_dataset"]
    train_df = df[df[cfg["data"]["dataset_col"]] == cd["train_dataset"]].copy()
    test_df  = df[df[cfg["data"]["dataset_col"]] == cd["test_dataset"]].copy()

    run_dir = init_run_dir(Path(cfg["outputs"]["root_dir"]), cfg["outputs"]["experiment_name"])
    write_json(run_dir / "environment.json", environment_snapshot())
    style = PlotStyle(dpi=1000)

    # single fold: split train into train/val subject-wise
    from src.data.splits import make_subjectwise_folds
    splits = make_subjectwise_folds(
        train_df.reset_index(drop=True),
        subject_col=cfg["data"]["subject_col"],
        label_col=cfg["data"]["label_col"],
        n_splits=5,  # internal CV for model selection; use first fold by default
        val_frac=float(cfg["cv"]["val_frac"]),
        shuffle=True,
        seed=seed,
        stratify_by_label=True,
    )
    sp = splits[0]
    fold_dir = init_fold_dir(run_dir, 0)
    save_config(fold_dir, cfg)

    fcfg = cfg["features"]
    feat_cfg = FeatureConfig(
        sample_rate=int(fcfg["sample_rate"]),
        highpass_hz=fcfg.get("highpass_hz", 50),
        n_fft=int(fcfg["n_fft"]),
        hop_length=int(fcfg["hop_length"]),
        n_mels=int(fcfg["n_mels"]),
        fmin=int(fcfg["fmin"]),
        fmax=int(fcfg["fmax"]),
        log_mel=bool(fcfg["log_mel"]),
        normalize=str(fcfg["normalize"]),
        max_len_sec=float(fcfg["max_len_sec"]),
    )

    if feat_cfg.normalize == "train_stats":
        tmp_ds = LungSoundDataset(train_df.reset_index(drop=True), sp.train_idx, feat_cfg, normalize_mode="none")
        tmp_loader = DataLoader(tmp_ds, batch_size=32, shuffle=False, num_workers=0)
        mels=[]
        for b in tmp_loader:
            mels.append(b["x"].numpy()[:,0,:,:])
        mels=np.concatenate(mels, axis=0)
        mean,std = compute_train_stats(mels)
    else:
        mean,std = None,None

    train_ds = LungSoundDataset(train_df.reset_index(drop=True), sp.train_idx, feat_cfg, normalize_mode=feat_cfg.normalize, train_mean=mean, train_std=std)
    val_ds   = LungSoundDataset(train_df.reset_index(drop=True), sp.val_idx, feat_cfg, normalize_mode=feat_cfg.normalize, train_mean=mean, train_std=std)
    test_ds  = LungSoundDataset(test_df.reset_index(drop=True), np.arange(len(test_df)), feat_cfg, normalize_mode=feat_cfg.normalize, train_mean=mean, train_std=std)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["project"]["num_workers"]))
    val_loader   = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["project"]["num_workers"]))
    test_loader  = DataLoader(test_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["project"]["num_workers"]))

    y = train_df.reset_index(drop=True).loc[sp.train_idx, cfg["data"]["label_col"]].values
    pos = (y==1).sum(); neg=(y==0).sum()
    pos_weight = float(neg/max(1,pos)) if pos>0 else None

    model = build_model(cfg)
    save_params_flops(model, fold_dir / "params_flops.csv", input_shape=(1,1,feat_cfg.n_mels, 200))

    tr = train_one_fold(
        model, train_loader, val_loader, device,
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        pos_weight=pos_weight,
        scheduler_cfg=cfg["train"],
        amp=bool(cfg["train"]["amp"]),
        aug_cfg=cfg.get("augment",{}),
        threshold=float(cfg["train"]["threshold"]),
        save_dir=fold_dir,
    )

    curves_path = save_curves(fold_dir, tr.curves)
    plot_learning_curves(curves_path, fold_dir / "learning_curve.png", style, title=f"Cross {cd['train_dataset']}→{cd['test_dataset']}")

    ck = torch.load(tr.best_ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device)
    metrics = evaluate_model(model, test_loader, device, threshold=float(cfg["train"]["threshold"]), out_dir=fold_dir, fold=0, split_name="test")
    pred = pd.read_csv(fold_dir / "predictions.csv")
    save_confusion(pred["y_true"].values.astype(int), pred["y_pred"].values.astype(int), fold_dir, style)
    save_metrics(fold_dir, metrics)

    print("DONE. Outputs in:", run_dir)

if __name__ == "__main__":
    main()
