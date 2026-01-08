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
import shutil
import pandas as pd
from src.utils.plotting import PlotStyle, plot_roc_pr_overlay, plot_learning_curves
from src.eval.confusion import save_confusion

def latest_run_dir(exp_dir: Path) -> Path | None:
    runs = sorted(exp_dir.glob("*"))
    return runs[-1] if runs else None

def make_figs_for_run(run: Path, fig_dir: Path, style: PlotStyle, title_prefix: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    roc_csvs = sorted(run.glob("fold_*/roc_curve.csv"))
    pr_csvs  = sorted(run.glob("fold_*/pr_curve.csv"))
    if roc_csvs and pr_csvs:
        plot_roc_pr_overlay(
            roc_csvs, pr_csvs,
            fig_dir / "Fig5_ROC_overlay.png",
            fig_dir / "Fig5_PR_overlay.png",
            style,
            title_prefix=title_prefix,
            out_mean_dir=fig_dir,
        )
    for f in sorted(run.glob("fold_*")):
        c = f / "curves_train_val.csv"
        if c.exists():
            plot_learning_curves(c, fig_dir / f"{f.name}_Fig4_learning_curve.png", style, title=f"{title_prefix}{f.name}")

        # confusion matrices: regenerate from predictions to ensure latest styling
        pred_csv = f / "predictions.csv"
        if pred_csv.exists():
            pred = pd.read_csv(pred_csv)
            save_confusion(
                pred["y_true"].values.astype(int),
                pred["y_pred"].values.astype(int),
                f,
                style,
            )
        cm_png = f / "confusion_matrix.png"
        if cm_png.exists():
            shutil.copy2(cm_png, fig_dir / f"{f.name}_confusion_matrix.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", type=str, default="outputs")
    ap.add_argument("--out_dir", type=str, default="paper_assets")
    ap.add_argument("--dpi", type=int, default=1000)
    args = ap.parse_args()

    outputs = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style = PlotStyle(dpi=args.dpi)

    for exp in sorted(outputs.glob("*")):
        if not exp.is_dir():
            continue
        run = latest_run_dir(exp)
        if run is None:
            continue

        # standard case
        make_figs_for_run(run, out_dir / "figures" / exp.name / "main", style, title_prefix=f"{exp.name} ")

        # nested baselines: create overlays per baseline
        fold0 = sorted(run.glob("fold_*"))
        if fold0:
            model_dirs = [p for p in fold0[0].iterdir() if p.is_dir()]
            for md in model_dirs:
                roc_csvs = sorted(run.glob(f"fold_*/{md.name}/roc_curve.csv"))
                pr_csvs  = sorted(run.glob(f"fold_*/{md.name}/pr_curve.csv"))
                fig_dir = out_dir / "figures" / exp.name / md.name
                if roc_csvs and pr_csvs:
                    plot_roc_pr_overlay(
                        roc_csvs,
                        pr_csvs,
                        fig_dir / "Fig5_ROC_overlay.png",
                        fig_dir / "Fig5_PR_overlay.png",
                        style,
                        title_prefix=f"{exp.name}:{md.name} ",
                        out_mean_dir=fig_dir,
                    )
                # learning curves
                for f in sorted(run.glob("fold_*")):
                    c = f / md.name / "curves_train_val.csv"
                    if c.exists():
                        plot_learning_curves(c, fig_dir / f"{f.name}_Fig4_learning_curve.png", style, title=f"{exp.name}:{md.name} {f.name}")

                    pred_csv = f / md.name / "predictions.csv"
                    if pred_csv.exists():
                        pred = pd.read_csv(pred_csv)
                        save_confusion(
                            pred["y_true"].values.astype(int),
                            pred["y_pred"].values.astype(int),
                            f / md.name,
                            style,
                        )
                    cm_png = f / md.name / "confusion_matrix.png"
                    if cm_png.exists():
                        shutil.copy2(cm_png, fig_dir / f"{f.name}_confusion_matrix.png")

    print("Figures written to:", (out_dir / "figures"))

if __name__ == "__main__":
    main()


# from __future__ import annotations
#
# # ---- path bootstrap: allow `from src...` when running as `python scripts\x.py`
# import sys as _sys
# from pathlib import Path as _Path
# _ROOT = _Path(__file__).resolve().parents[1]
# if str(_ROOT) not in _sys.path:
#     _sys.path.insert(0, str(_ROOT))
# # ---- end bootstrap
#
# import argparse
# from pathlib import Path
# from src.utils.plotting import PlotStyle, plot_roc_pr_overlay, plot_learning_curves
#
# def latest_run_dir(exp_dir: Path) -> Path | None:
#     runs = sorted(exp_dir.glob("*"))
#     return runs[-1] if runs else None
#
# def make_figs_for_run(run: Path, fig_dir: Path, style: PlotStyle, title_prefix: str):
#     fig_dir.mkdir(parents=True, exist_ok=True)
#     roc_csvs = sorted(run.glob("fold_*/roc_curve.csv"))
#     pr_csvs  = sorted(run.glob("fold_*/pr_curve.csv"))
#     if roc_csvs and pr_csvs:
#         plot_roc_pr_overlay(
#             roc_csvs, pr_csvs,
#             fig_dir / "Fig5_ROC_overlay.png",
#             fig_dir / "Fig5_PR_overlay.png",
#             style,
#             title_prefix=title_prefix
#         )
#     for f in sorted(run.glob("fold_*")):
#         c = f / "curves_train_val.csv"
#         if c.exists():
#             plot_learning_curves(c, fig_dir / f"{f.name}_Fig4_learning_curve.png", style, title=f"{title_prefix}{f.name}")
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--outputs_dir", type=str, default="outputs")
#     ap.add_argument("--out_dir", type=str, default="paper_assets")
#     ap.add_argument("--dpi", type=int, default=1000)
#     args = ap.parse_args()
#
#     outputs = Path(args.outputs_dir)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     style = PlotStyle(dpi=args.dpi)
#
#     for exp in sorted(outputs.glob("*")):
#         if not exp.is_dir():
#             continue
#         run = latest_run_dir(exp)
#         if run is None:
#             continue
#
#         # standard case
#         make_figs_for_run(run, out_dir / "figures" / exp.name / "main", style, title_prefix=f"{exp.name} ")
#
#         # nested baselines: create overlays per baseline
#         fold0 = sorted(run.glob("fold_*"))
#         if fold0:
#             model_dirs = [p for p in fold0[0].iterdir() if p.is_dir()]
#             for md in model_dirs:
#                 roc_csvs = sorted(run.glob(f"fold_*/{md.name}/roc_curve.csv"))
#                 pr_csvs  = sorted(run.glob(f"fold_*/{md.name}/pr_curve.csv"))
#                 fig_dir = out_dir / "figures" / exp.name / md.name
#                 if roc_csvs and pr_csvs:
#                     plot_roc_pr_overlay(roc_csvs, pr_csvs, fig_dir / "Fig5_ROC_overlay.png", fig_dir / "Fig5_PR_overlay.png", style, title_prefix=f"{exp.name}:{md.name} ")
#                 # learning curves
#                 for f in sorted(run.glob("fold_*")):
#                     c = f / md.name / "curves_train_val.csv"
#                     if c.exists():
#                         plot_learning_curves(c, fig_dir / f"{f.name}_Fig4_learning_curve.png", style, title=f"{exp.name}:{md.name} {f.name}")
#
#     print("Figures written to:", (out_dir / "figures"))
#
# if __name__ == "__main__":
#     main()
