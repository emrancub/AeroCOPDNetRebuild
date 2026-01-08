from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..utils.plotting import apply_publication_style, PlotStyle, savefig

def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, style: PlotStyle) -> Path:
    """Saves publication-ready confusion matrices.

    Outputs:
      - confusion_matrix.csv (counts)
      - confusion_matrix_row_norm.csv (row-normalized, 0..1)
      - confusion_matrix_total_norm.csv (total-normalized, 0..1)
      - confusion_matrix.png (counts + *total-%* per cell)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])
    df.to_csv(out_dir / "confusion_matrix.csv", index=True)

    row_sum = cm.sum(axis=1, keepdims=True) + 1e-9
    cm_row = cm / row_sum
    pd.DataFrame(cm_row, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(out_dir / "confusion_matrix_row_norm.csv", index=True)

    tot = cm.sum() + 1e-9
    cm_tot = cm / tot
    pd.DataFrame(cm_tot, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(out_dir / "confusion_matrix_total_norm.csv", index=True)

    apply_publication_style(style)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    # Show total-normalized background so percentages match the annotation.
    ax.imshow(cm_tot, interpolation="nearest")
    ax.set_title("Confusion Matrix (count + % of total)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-COPD (0)","COPD (1)"])
    ax.set_yticklabels(["Non-COPD (0)","COPD (1)"])

    for (i,j), v in np.ndenumerate(cm):
        pct = 100.0 * cm_tot[i,j]
        ax.text(j, i, f"{v}\n{pct:.1f}%", ha="center", va="center", fontweight="bold")
    savefig(out_dir / "confusion_matrix.png", style)
    return out_dir / "confusion_matrix.png"


# from __future__ import annotations
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from ..utils.plotting import apply_publication_style, PlotStyle, savefig
#
# def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, style: PlotStyle) -> Path:
#     """Saves:
#     - confusion_matrix.csv (counts)
#     - confusion_matrix_row_norm.csv (row-normalized percentages)
#     - confusion_matrix.png (shows both counts and row-% in each cell)
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     cm = confusion_matrix(y_true, y_pred, labels=[0,1])
#     df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])
#     df.to_csv(out_dir / "confusion_matrix.csv", index=True)
#
#     row_sum = cm.sum(axis=1, keepdims=True) + 1e-9
#     cm_row = cm / row_sum
#     pd.DataFrame(cm_row, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(out_dir / "confusion_matrix_row_norm.csv", index=True)
#
#     apply_publication_style(style)
#     fig = plt.figure(figsize=(7,6))
#     ax = fig.add_subplot(111)
#     ax.imshow(cm_row, interpolation="nearest")  # show normalized background
#     ax.set_title("Confusion Matrix (row-normalized)")
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(["Non-COPD (0)","COPD (1)"])
#     ax.set_yticklabels(["Non-COPD (0)","COPD (1)"])
#
#     for (i,j), v in np.ndenumerate(cm):
#         pct = 100.0 * cm_row[i,j]
#         ax.text(j, i, f"{v}\n{pct:.1f}%", ha="center", va="center", fontweight="bold")
#     savefig(out_dir / "confusion_matrix.png", style)
#     return out_dir / "confusion_matrix.png"


