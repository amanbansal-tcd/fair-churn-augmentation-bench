"""
evaluation.py
=============
Classification metrics (9 metrics) and diagnostic visualisations.

Figures produced
----------------
  Fig 6 — ROC curves, PR curves, metric bar chart
  Fig 7 — Confusion matrices for all baseline models
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src.config import MODEL_COLORS, FIG_DIR, FIG_DPI, REPORT_DIR

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_prob) -> dict:
    """
    Compute all nine classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc,
                    bal_acc, log_loss, mcc, y_pred, y_prob
    """
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
        "bal_acc":   balanced_accuracy_score(y_true, y_pred),
        "log_loss":  log_loss(y_true, y_prob),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "y_pred":    np.asarray(y_pred),
        "y_prob":    np.asarray(y_prob),
    }


def metrics_dataframe(results: dict) -> pd.DataFrame:
    """
    Convert a results dict to a tidy DataFrame sorted by ROC-AUC.

    Parameters
    ----------
    results : dict
        {model_name: metrics_dict} as returned by train_all().
    """
    metric_cols = [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "bal_acc", "log_loss", "mcc",
    ]
    rows = [
        {"Model": name, **{c: round(res[c], 4) for c in metric_cols}}
        for name, res in results.items()
    ]
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def plot_diagnostics(
    results: dict,
    y_test,
    out_dir: str | Path | None = None,
) -> None:
    """
    Save Fig 6 (ROC/PR/bar) and Fig 7 (confusion matrices).

    Parameters
    ----------
    results : dict
        Output of train_all().
    y_test : array-like
        True labels.
    out_dir : str | Path | None
        Directory for PNG files; defaults to FIG_DIR.
    """
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    _fig6_curves(results, y_test, out)
    _fig7_confusion(results, y_test, out)

    # Save CSV report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dataframe(results).to_csv(REPORT_DIR / "baseline_metrics.csv", index=False)
    logger.info("Baseline metrics CSV saved.")


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _fig6_curves(results, y_test, out):
    """ROC curves, PR curves, and grouped metric bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("Fig 6 | Baseline Model Performance Comparison",
                 fontsize=14, fontweight="bold")

    # ROC curves
    ax = axes[0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, "grey"), lw=2,
                label=f"{name}  AUC={res['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # PR curves
    ax = axes[1]
    for name, res in results.items():
        p, r, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(r, p, color=MODEL_COLORS.get(name, "grey"), lw=2,
                label=f"{name}  PR={res['pr_auc']:.4f}")
    ax.axhline(np.asarray(y_test).mean(), color="black", ls="--",
               alpha=0.5, lw=1, label="Chance")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Grouped metric bar chart
    ax      = axes[2]
    metrics = ["roc_auc", "f1", "bal_acc", "pr_auc"]
    m_lbls  = ["ROC-AUC", "F1", "Bal.Acc", "PR-AUC"]
    x       = np.arange(len(metrics))
    w       = 0.15
    for i, (name, res) in enumerate(results.items()):
        ax.bar(x + i * w, [res[m] for m in metrics], w,
               label=name, color=MODEL_COLORS.get(name, "grey"), alpha=0.85)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(m_lbls)
    ax.set(title="Metric Comparison", ylabel="Score", ylim=(0, 1))
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, out / "fig06_roc_pr.png")


def _fig7_confusion(results, y_test, out):
    """One confusion matrix per model."""
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Fig 7 | Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["NO", "YES"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(
            f"{name}\nF1={res['f1']:.3f}  AUC={res['roc_auc']:.3f}",
            fontsize=9, fontweight="bold",
        )
    plt.tight_layout()
    _save(fig, out / "fig07_confusion.png")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
