from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotStyle:
    """Centralized plotting style to enforce publication-ready output."""

    dpi: int = 1000
    base_font: int = 18
    title_font: int = 24
    lw: float = 2.6
    axis_lw: float = 2.2
    marker_size: float = 6.0


def apply_publication_style(style: PlotStyle) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "font.size": style.base_font,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": style.axis_lw,
            "lines.linewidth": style.lw,
            "xtick.major.width": style.axis_lw,
            "ytick.major.width": style.axis_lw,
            "xtick.labelsize": style.base_font,
            "ytick.labelsize": style.base_font,
            "legend.fontsize": style.base_font,
            "legend.frameon": True,
        }
    )


def savefig(path: Path, style: PlotStyle, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=style.dpi, bbox_inches="tight")
    plt.close()


def _safe_interp_mean(xy_list: list[tuple[np.ndarray, np.ndarray]], x_grid: np.ndarray) -> np.ndarray:
    """Interpolate y over x_grid for each curve and average.

    Assumes x is monotonic increasing after sorting.
    """
    ys = []
    for x, y in xy_list:
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        # sort and unique
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        # ensure endpoints
        if x[0] > x_grid[0]:
            x = np.concatenate([[x_grid[0]], x])
            y = np.concatenate([[y[0]], y])
        if x[-1] < x_grid[-1]:
            x = np.concatenate([x, [x_grid[-1]]])
            y = np.concatenate([y, [y[-1]]])
        # remove duplicate x for np.interp
        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]
        ys.append(np.interp(x_grid, x_u, y_u))
    return np.mean(np.stack(ys, axis=0), axis=0)


def plot_learning_curves(curves_csv: Path, out_png: Path, style: PlotStyle, title: str) -> None:
    """Save loss + accuracy curves.

    Produces:
      - out_png (2-panel: loss + accuracy)
      - out_png with suffix _loss.png and _acc.png
    """
    apply_publication_style(style)
    df = pd.read_csv(curves_csv)

    # 2-panel combined
    fig = plt.figure(figsize=(11, 9))
    ax1 = fig.add_subplot(211)
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax1.plot(df["epoch"], df["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss", fontsize=style.title_font, fontweight="bold")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(212)
    if "train_accuracy" in df.columns and "val_accuracy" in df.columns:
        ax2.plot(df["epoch"], df["train_accuracy"], label="Train Acc")
        ax2.plot(df["epoch"], df["val_accuracy"], label="Val Acc")
        ax2.set_ylabel("Accuracy")
    else:
        ax2.text(
            0.5,
            0.5,
            "Accuracy not logged",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontweight="bold",
        )
    ax2.set_xlabel("Epoch")
    ax2.set_title(f"{title} - Accuracy", fontsize=style.title_font, fontweight="bold")
    ax2.legend(loc="best")
    savefig(out_png, style)

    # Separate loss
    loss_png = out_png.with_name(out_png.stem + "_loss.png")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} - Loss", fontsize=style.title_font, fontweight="bold")
    ax.legend(loc="best")
    savefig(loss_png, style)

    # Separate accuracy
    acc_png = out_png.with_name(out_png.stem + "_acc.png")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    if "train_accuracy" in df.columns and "val_accuracy" in df.columns:
        ax.plot(df["epoch"], df["train_accuracy"], label="Train Acc")
        ax.plot(df["epoch"], df["val_accuracy"], label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Accuracy", fontsize=style.title_font, fontweight="bold")
    ax.legend(loc="best")
    savefig(acc_png, style)


def plot_roc_pr_overlay(
    roc_csvs: Iterable[Path],
    pr_csvs: Iterable[Path],
    out_roc_png: Path,
    out_pr_png: Path,
    style: PlotStyle,
    title_prefix: str = "",
    out_mean_dir: Optional[Path] = None,
) -> None:
    """Overlay ROC/PR curves for all folds + mean curve.

    - ROC legend: bottom-right
    - PR legend: bottom-left
    - Saves mean curve points to CSV if out_mean_dir is provided.
    """
    apply_publication_style(style)

    roc_csvs = list(roc_csvs)
    pr_csvs = list(pr_csvs)

    # -------- ROC --------
    roc_xy = []
    roc_labels = []
    for i, p in enumerate(sorted(roc_csvs)):
        df = pd.read_csv(p)
        fpr = df["fpr"].to_numpy(dtype=float)
        tpr = df["tpr"].to_numpy(dtype=float)
        roc_xy.append((fpr, tpr))
        roc_labels.append(f"Fold {i+1}")

    x_grid = np.linspace(0.0, 1.0, 1001)
    mean_tpr = _safe_interp_mean(roc_xy, x_grid) if roc_xy else None

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111)
    for (fpr, tpr), lab in zip(roc_xy, roc_labels):
        ax.plot(fpr, tpr, alpha=0.85, label=lab)

    if mean_tpr is not None:
        ax.plot(
            x_grid,
            mean_tpr,
            linestyle="-",
            linewidth=style.lw + 1.2,
            color="black",
            label="Mean",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=style.lw, color="0.4", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix}ROC Curves (per fold)", fontsize=style.title_font, fontweight="bold")
    ax.legend(loc="lower right")
    savefig(out_roc_png, style)

    if out_mean_dir is not None and mean_tpr is not None:
        out_mean_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"fpr": x_grid, "tpr_mean": mean_tpr}).to_csv(out_mean_dir / "roc_mean_curve.csv", index=False)

    # -------- PR --------
    pr_xy = []
    pr_labels = []
    for i, p in enumerate(sorted(pr_csvs)):
        df = pd.read_csv(p)
        recall = df["recall"].to_numpy(dtype=float)
        precision = df["precision"].to_numpy(dtype=float)
        pr_xy.append((recall, precision))
        pr_labels.append(f"Fold {i+1}")

    r_grid = np.linspace(0.0, 1.0, 1001)
    mean_prec = _safe_interp_mean(pr_xy, r_grid) if pr_xy else None
    mean_prec = np.clip(mean_prec, 0.0, 1.0) if mean_prec is not None else None

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111)
    for (rec, pre), lab in zip(pr_xy, pr_labels):
        ax.plot(rec, pre, alpha=0.85, label=lab)

    if mean_prec is not None:
        ax.plot(
            r_grid,
            mean_prec,
            linestyle="-",
            linewidth=style.lw + 1.2,
            color="black",
            label="Mean",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix}PR Curves (per fold)", fontsize=style.title_font, fontweight="bold")
    ax.legend(loc="lower left")
    savefig(out_pr_png, style)

    if out_mean_dir is not None and mean_prec is not None:
        out_mean_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"recall": r_grid, "precision_mean": mean_prec}).to_csv(out_mean_dir / "pr_mean_curve.csv", index=False)



# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, Tuple, Dict, Any
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import pandas as pd
#
# @dataclass
# class PlotStyle:
#     dpi: int = 1000
#     base_font: int = 18
#     title_font: int = 24
#     lw: float = 2.4
#     axis_lw: float = 2.2
#
# def apply_publication_style(style: PlotStyle) -> None:
#     plt.rcParams.update({
#         "figure.dpi": style.dpi,
#         "savefig.dpi": style.dpi,
#         "font.size": style.base_font,
#         "font.weight": "bold",
#         "axes.labelweight": "bold",
#         "axes.titleweight": "bold",
#         "axes.linewidth": style.axis_lw,
#         "lines.linewidth": style.lw,
#         "xtick.major.width": style.axis_lw,
#         "ytick.major.width": style.axis_lw,
#         "xtick.labelsize": style.base_font,
#         "ytick.labelsize": style.base_font,
#         "legend.fontsize": style.base_font,
#     })
#
# def savefig(path: Path, style: PlotStyle, tight: bool = True) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     if tight:
#         plt.tight_layout()
#     plt.savefig(path, dpi=style.dpi, bbox_inches="tight")
#     plt.close()
#
# def plot_learning_curves(curves_csv: Path, out_png: Path, style: PlotStyle, title: str) -> None:
#     apply_publication_style(style)
#     df = pd.read_csv(curves_csv)
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111)
#     ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
#     ax.plot(df["epoch"], df["val_loss"], label="Val Loss")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss")
#     ax.set_title(title)
#     ax.legend()
#     savefig(out_png, style)
#
# def plot_roc_pr_overlay(roc_csvs, pr_csvs, out_roc_png: Path, out_pr_png: Path, style: PlotStyle, title_prefix: str=""):
#     apply_publication_style(style)
#     # ROC
#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111)
#     for p in roc_csvs:
#         df = pd.read_csv(p)
#         ax.plot(df["fpr"], df["tpr"], alpha=0.85)
#     ax.plot([0,1],[0,1], linestyle="--")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title(f"{title_prefix}ROC Curves (per fold)")
#     savefig(out_roc_png, style)
#     # PR
#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111)
#     for p in pr_csvs:
#         df = pd.read_csv(p)
#         ax.plot(df["recall"], df["precision"], alpha=0.85)
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")
#     ax.set_title(f"{title_prefix}PR Curves (per fold)")
#     savefig(out_pr_png, style)
