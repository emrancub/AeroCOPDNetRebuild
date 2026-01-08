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
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.io import read_yaml, write_yaml, sha1_of_file, save_df_csv
from src.utils.repro import set_global_seed, get_device, environment_snapshot
from src.utils.logging import init_run_dir, init_fold_dir, save_config, save_curves, save_metrics
from src.data.splits import make_subjectwise_folds
from src.data.features import FeatureConfig, compute_train_stats
from src.data.loaders_icbhi import LungSoundDataset
from src.train.trainer import train_one_fold
from src.eval.evaluate import evaluate_model
from src.eval.flops_params import save_params_flops
from src.eval.confusion import save_confusion
from src.utils.plotting import PlotStyle, plot_learning_curves

from src.models.aerocpdnet import AeroCOPDNet, AeroCOPDNetCfg
from src.models.baselines import BasicCNN, CRNN, LSTMNet, GRUNet
from src.models.mobilenet import MobileNetV2Spectrogram

def load_config(path: Path) -> dict:
    cfg = read_yaml(path)
    if "inherit" in cfg:
        base = read_yaml(path.parent / cfg["inherit"])
        # shallow merge
        merged = copy.deepcopy(base)
        def merge(a,b):
            for k,v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k]=v
        merge(merged, cfg)
        return merged
    return cfg

def build_model(cfg: dict):
    name = cfg["train"]["model"]
    if name == "aerocpdnet":
        ac = cfg["train"].get("aerocpdnet", {})
        mcfg = AeroCOPDNetCfg(
            use_se=bool(ac.get("use_se", True)),
            pooling=str(ac.get("pooling", "gap")),
            base_channels=int(ac.get("base_channels", 32)),
        )
        return AeroCOPDNet(mcfg)
    if name == "basic_cnn":
        return BasicCNN()
    if name == "crnn":
        return CRNN()
    if name == "lstm":
        return LSTMNet()
    if name == "gru":
        return GRUNet()
    if name == "mobilenetv2":
        return MobileNetV2Spectrogram()
    raise ValueError(f"Unknown model: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    seed = int(args.seed) if args.seed is not None else int(cfg["project"]["seed"])
    set_global_seed(seed, deterministic=True)
    device = get_device(args.device if args.device is not None else cfg["project"]["device"])

    manifest = pd.read_csv(cfg["data"]["manifest_csv"])
    subject_col = cfg["data"]["subject_col"]
    label_col = cfg["data"]["label_col"]

    splits = make_subjectwise_folds(
        manifest,
        subject_col=subject_col,
        label_col=label_col,
        n_splits=int(cfg["cv"]["n_splits"]),
        val_frac=float(cfg["cv"]["val_frac"]),
        shuffle=bool(cfg["cv"]["shuffle"]),
        seed=seed,
        stratify_by_label=bool(cfg["cv"]["stratify_by_label"]),
    )

    run_dir = init_run_dir(Path(cfg["outputs"]["root_dir"]), cfg["outputs"]["experiment_name"])
    # environment snapshot
    from src.utils.io import write_json
    write_json(run_dir / "environment.json", environment_snapshot())

    style = PlotStyle(dpi=1000)

    all_fold_metrics = []

    for sp in splits:
        fold_dir = init_fold_dir(run_dir, sp.fold)
        save_config(fold_dir, cfg)

        # Leakage audit: save subjects per split + file hashes
        train_subj = sorted(set(manifest.loc[sp.train_idx, subject_col].astype(str)))
        val_subj   = sorted(set(manifest.loc[sp.val_idx, subject_col].astype(str)))
        test_subj  = sorted(set(manifest.loc[sp.test_idx, subject_col].astype(str)))
        pd.DataFrame({
            "split": ["train"]*len(train_subj) + ["val"]*len(val_subj) + ["test"]*len(test_subj),
            "subject_id": train_subj + val_subj + test_subj,
        }).to_csv(fold_dir / "split_subjects.csv", index=False)
        # file sha1 list (by split)
        sha_rows=[]
        for split_name, idxs in [("train", sp.train_idx), ("val", sp.val_idx), ("test", sp.test_idx)]:
            for p in manifest.loc[idxs, "filepath"].astype(str).unique():
                try:
                    sha_rows.append({"split": split_name, "filepath": p, "sha1": sha1_of_file(Path(p))})
                except Exception:
                    sha_rows.append({"split": split_name, "filepath": p, "sha1": "NA"})
        pd.DataFrame(sha_rows).to_csv(fold_dir / "split_files_sha1.csv", index=False)

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

        # Compute train mel stats (for normalize=train_stats)
        # For speed, we compute stats on a subset if huge; here compute on full train set.
        if feat_cfg.normalize == "train_stats":
            tmp_ds = LungSoundDataset(manifest, sp.train_idx, feat_cfg, normalize_mode="none")
            tmp_loader = DataLoader(tmp_ds, batch_size=32, shuffle=False, num_workers=0)
            mels=[]
            for b in tmp_loader:
                mels.append(b["x"].numpy()[:,0,:,:])  # [B,F,T]
            mels = np.concatenate(mels, axis=0)
            mean, std = compute_train_stats(mels)
        else:
            mean, std = None, None

        train_ds = LungSoundDataset(manifest, sp.train_idx, feat_cfg, normalize_mode=feat_cfg.normalize, train_mean=mean, train_std=std)
        val_ds   = LungSoundDataset(manifest, sp.val_idx, feat_cfg, normalize_mode=feat_cfg.normalize, train_mean=mean, train_std=std)
        test_ds  = LungSoundDataset(manifest, sp.test_idx, feat_cfg, normalize_mode=feat_cfg.normalize, train_mean=mean, train_std=std)

        train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                                  num_workers=int(cfg["project"]["num_workers"]))
        val_loader   = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                                  num_workers=int(cfg["project"]["num_workers"]))
        test_loader  = DataLoader(test_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                                  num_workers=int(cfg["project"]["num_workers"]))

        # pos_weight
        pos_weight = None
        if cfg["train"]["pos_weight"] == "auto":
            y = manifest.loc[sp.train_idx, label_col].values
            pos = (y==1).sum(); neg = (y==0).sum()
            if pos > 0:
                pos_weight = float(neg / max(1,pos))
        else:
            pos_weight = float(cfg["train"]["pos_weight"])

        # model
        model_name = cfg["train"]["model"]
        if model_name == "baselines":
            # run each baseline sequentially into subfolders
            for base in cfg["train"].get("baselines", ["basic_cnn","crnn","lstm","gru"]):
                sub_cfg = copy.deepcopy(cfg)
                sub_cfg["train"]["model"] = base
                subdir = fold_dir / base
                subdir.mkdir(parents=True, exist_ok=True)
                m = build_model(sub_cfg)
                save_params_flops(m, subdir / "params_flops.csv", input_shape=(1,1,feat_cfg.n_mels, 200))
                tr = train_one_fold(
                    m, train_loader, val_loader, device,
                    epochs=int(sub_cfg["train"]["epochs"]),
                    lr=float(sub_cfg["train"]["lr"]),
                    weight_decay=float(sub_cfg["train"]["weight_decay"]),
                    pos_weight=pos_weight,
                    scheduler_cfg=sub_cfg["train"],
                    amp=bool(sub_cfg["train"]["amp"]),
                    aug_cfg=sub_cfg.get("augment",{}),
                    threshold=float(sub_cfg["train"]["threshold"]),
                    save_dir=subdir,
                )
                curves_path = save_curves(subdir, tr.curves)
                plot_learning_curves(curves_path, subdir / "learning_curve.png", style, title=f"{base} Fold {sp.fold}")
                # load best
                ck = torch.load(tr.best_ckpt, map_location=device)
                m.load_state_dict(ck["model"])
                m.to(device)
                metrics = evaluate_model(m, test_loader, device, threshold=float(sub_cfg["train"]["threshold"]), out_dir=subdir, fold=sp.fold, split_name="test")
                # confusion
                pred = pd.read_csv(subdir / "predictions.csv")
                save_confusion(pred["y_true"].values.astype(int), pred["y_pred"].values.astype(int), subdir, style)
                save_metrics(subdir, metrics)
            continue

        m = build_model(cfg)
        save_params_flops(m, fold_dir / "params_flops.csv", input_shape=(1,1,feat_cfg.n_mels, 200))

        tr = train_one_fold(
            m, train_loader, val_loader, device,
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
        plot_learning_curves(curves_path, fold_dir / "learning_curve.png", style, title=f"{cfg['train']['model']} Fold {sp.fold}")

        ck = torch.load(tr.best_ckpt, map_location=device)
        m.load_state_dict(ck["model"])
        m.to(device)
        metrics = evaluate_model(m, test_loader, device, threshold=float(cfg["train"]["threshold"]), out_dir=fold_dir, fold=sp.fold, split_name="test")
        # confusion
        pred = pd.read_csv(fold_dir / "predictions.csv")
        save_confusion(pred["y_true"].values.astype(int), pred["y_pred"].values.astype(int), fold_dir, style)
        save_metrics(fold_dir, metrics)
        all_fold_metrics.append({"fold": sp.fold, **metrics})

    # save fold summary
    if len(all_fold_metrics) > 0:
        df = pd.DataFrame(all_fold_metrics)
        df.to_csv(run_dir / "fold_metrics.csv", index=False)

    print("DONE. Outputs in:", run_dir)

if __name__ == "__main__":
    main()
