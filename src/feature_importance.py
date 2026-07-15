"""
feature_importance.py
=====================
Tree-based importance and SHAP analysis.

Figures produced
----------------
  Fig 8 — Gini importance for all four tree models
  Fig 9 — SHAP beeswarm + bar for LightGBM
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import FIG_DIR, FIG_DPI, FEATURE_LABELS, RANDOM_SEED

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def plot_feature_importance(
    models: dict,
    features: list[str],
    out_dir: str | Path | None = None,
) -> None:
    """
    Save Fig 8: Gini feature importance for the four tree-based models.

    Parameters
    ----------
    models : dict
        Fitted model dict (output of train_all before results — pass the
        models dict directly, not the results dict).
    features : list[str]
        Feature names in the order they were passed to fit().
    out_dir : str | Path | None
    """
    out    = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    labels = [FEATURE_LABELS.get(f, f) for f in features]
    tree_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Fig 8 | Feature Importance — Tree-Based Models",
                 fontsize=14, fontweight="bold")

    for idx, name in enumerate(tree_names):
        if name not in models:
            continue
        ax   = axes[idx // 2][idx % 2]
        imp  = models[name].feature_importances_
        sidx = np.argsort(imp)
        cols = plt.cm.viridis(np.linspace(0.25, 0.85, len(features)))
        ax.barh(range(len(features)), imp[sidx],
                color=cols, edgecolor="white")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([labels[i] for i in sidx], fontsize=9)
        ax.set_title(
            f"{name} — top: {labels[np.argmax(imp)]}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Importance")

    plt.tight_layout()
    _save(fig, out / "fig08_importance.png")


def plot_shap(
    model,
    X_test,
    features: list[str],
    out_dir: str | Path | None = None,
    n_sample: int = 600,
) -> None:
    """
    Save Fig 9: SHAP beeswarm and global bar chart for LightGBM.

    Handles SHAP output format changes across versions:
      - older SHAP : list [class0_array, class1_array] → use [1]
      - SHAP 0.52+ : 3-D ndarray [samples, features, classes] → use [:,:,1]
      - SHAP 0.52  : 2-D ndarray [samples, features] → use directly

    Parameters
    ----------
    model : fitted LightGBM model
    X_test : pd.DataFrame
    features : list[str]
    out_dir : str | Path | None
    n_sample : int
        Number of test samples used for SHAP (controls speed).
    """
    out    = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    logger.info("Computing SHAP values on %d samples...", n_sample)
    Xs  = X_test.sample(min(n_sample, len(X_test)), random_state=RANDOM_SEED)
    exp = shap.TreeExplainer(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = exp.shap_values(Xs)

    # Normalise SHAP output to a 2-D array for the positive class
    if isinstance(sv, list):
        sv = sv[1]                  # older SHAP: [class0, class1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]            # SHAP 0.52+: [samples, features, classes]
    # else: 2-D array — use as-is

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Fig 9 | SHAP Analysis — LightGBM",
                 fontsize=14, fontweight="bold")

    plt.sca(axes[0])
    shap.summary_plot(sv, Xs, feature_names=labels,
                      show=False, max_display=12, plot_type="dot")
    axes[0].set_title("SHAP Beeswarm", fontweight="bold")

    plt.sca(axes[1])
    shap.summary_plot(sv, Xs, feature_names=labels,
                      show=False, max_display=12, plot_type="bar")
    axes[1].set_title("SHAP Global Importance", fontweight="bold")

    plt.tight_layout()
    _save(fig, out / "fig09_shap.png")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
