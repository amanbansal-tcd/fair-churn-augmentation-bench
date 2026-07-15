"""
final_model.py
==============
Final debiased model (M12): LightGBM without Ethnicity, Language,
and Home Ownership.

Figures produced
----------------
  Fig 13 — Final model performance dashboard
  Fig 14 — Final SHAP analysis
  Fig 15 — Lift & gain charts
  Fig 16 — Calibration curve
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
import lightgbm as lgb

from src.config import (
    ALL_FEATURES, SENSITIVE_FEATS, LGB_FINAL_PARAMS,
    RANDOM_SEED, FIG_DIR, FIG_DPI, MODEL_DIR, FEATURE_LABELS,
)
from src.evaluation import evaluate_model

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Columns to drop for M12 (the recommended production model)
FINAL_DROP = (
    SENSITIVE_FEATS["ETHNICITY"] +
    SENSITIVE_FEATS["LANGUAGE"] +
    SENSITIVE_FEATS["HOMEOWN"]
)


def get_final_features(all_features: list[str] | None = None) -> list[str]:
    """Return the M12 feature list (ALL_FEATURES minus FINAL_DROP)."""
    return [f for f in (all_features or ALL_FEATURES) if f not in FINAL_DROP]


def train_final_model(
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple:
    """
    Train and evaluate the M12 final debiased model.

    Returns
    -------
    model : fitted LGBMClassifier
    metrics : dict
        All evaluation metrics plus 'features_used'.
    """
    feats = [f for f in get_final_features() if f in X_train.columns]
    logger.info("Final model (M12): %d features: %s", len(feats), feats)

    model = lgb.LGBMClassifier(class_weight="balanced", **LGB_FINAL_PARAMS)
    model.fit(X_train[feats], y_train)

    yp  = model.predict(X_test[feats])
    ypr = model.predict_proba(X_test[feats])[:, 1]

    metrics = evaluate_model(y_test, yp, ypr)
    metrics["features_used"] = feats

    logger.info(
        "M12: AUC=%.4f  F1=%.4f  BalAcc=%.4f",
        metrics["roc_auc"], metrics["f1"], metrics["bal_acc"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "final_model_m12.pkl")
    logger.info("Model saved → outputs/models/final_model_m12.pkl")

    return model, metrics


def plot_final_diagnostics(
    model,
    X_test,
    y_test,
    out_dir: str | Path | None = None,
    n_shap: int = 600,
) -> None:
    """
    Save Figs 13–16.

    Parameters
    ----------
    model : fitted LGBMClassifier (M12)
    X_test : pd.DataFrame
        Test set restricted to final features.
    y_test : array-like
    out_dir : str | Path | None
    n_shap : int
        Number of test samples used for SHAP computation.
    """
    out    = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    feats  = list(X_test.columns)
    labels = [FEATURE_LABELS.get(f, f) for f in feats]
    ypr    = model.predict_proba(X_test)[:, 1]
    yp     = model.predict(X_test)

    _fig13_dashboard(model, X_test, y_test, yp, ypr, feats, labels, out)
    _fig14_shap(model, X_test, labels, out, n_shap)
    _fig15_lift(y_test, ypr, out)
    _fig16_calibration(y_test, ypr, out)


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _fig13_dashboard(model, X_test, y_test, yp, ypr, feats, labels, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Fig 13 | Final Debiased Model (M12) — Performance Dashboard",
                 fontsize=14, fontweight="bold")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, ypr)
    axes[0][0].plot(fpr, tpr, "#8e44ad", lw=2.5,
                    label=f"AUC = {roc_auc_score(y_test, ypr):.4f}")
    axes[0][0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0][0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0][0].legend(); axes[0][0].grid(alpha=0.3)

    # PR curve
    p, r, _ = precision_recall_curve(y_test, ypr)
    axes[0][1].plot(r, p, "#8e44ad", lw=2.5)
    axes[0][1].axhline(np.asarray(y_test).mean(), color="black",
                       ls="--", alpha=0.5, label="Chance")
    axes[0][1].set(xlabel="Recall", ylabel="Precision", title="PR Curve")
    axes[0][1].legend(); axes[0][1].grid(alpha=0.3)

    # Confusion matrix
    cm = confusion_matrix(y_test, yp)
    ConfusionMatrixDisplay(cm, display_labels=["NO", "YES"]).plot(
        ax=axes[1][0], colorbar=False, cmap="Purples"
    )
    axes[1][0].set_title(
        f"Confusion Matrix — F1 = {f1_score(y_test, yp):.4f}"
    )

    # Feature importance
    imp  = model.feature_importances_
    sidx = np.argsort(imp)
    cols = plt.cm.plasma(np.linspace(0.2, 0.85, len(feats)))
    axes[1][1].barh(range(len(feats)), imp[sidx],
                    color=cols, edgecolor="white")
    axes[1][1].set_yticks(range(len(feats)))
    axes[1][1].set_yticklabels([labels[i] for i in sidx], fontsize=9)
    axes[1][1].set(title="Feature Importance (M12)", xlabel="Score")

    plt.tight_layout()
    _save(fig, out / "fig13_final_dashboard.png")


def _fig14_shap(model, X_test, labels, out, n_shap):
    logger.info("Computing SHAP for final model (%d samples)...", n_shap)
    Xs  = X_test.sample(min(n_shap, len(X_test)), random_state=RANDOM_SEED)
    exp = shap.TreeExplainer(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = exp.shap_values(Xs)

    # Normalise output format across SHAP versions
    if isinstance(sv, list):
        sv = sv[1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Fig 14 | Final Model SHAP Analysis (M12)",
                 fontsize=14, fontweight="bold")
    plt.sca(axes[0])
    shap.summary_plot(sv, Xs, feature_names=labels,
                      show=False, max_display=9, plot_type="dot")
    axes[0].set_title("SHAP Beeswarm", fontweight="bold")
    plt.sca(axes[1])
    shap.summary_plot(sv, Xs, feature_names=labels,
                      show=False, max_display=9, plot_type="bar")
    axes[1].set_title("SHAP Global Importance", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "fig14_final_shap.png")


def _fig15_lift(y_test, ypr, out):
    ya   = np.asarray(y_test)
    idx  = np.argsort(-ypr)
    ys   = ya[idx]
    n    = len(ys)
    pos  = ys.sum()
    gain = np.cumsum(ys) / pos
    pct  = np.arange(1, n + 1) / n
    lift = gain / pct

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fig 15 | Lift & Gain Charts — Final Model (M12)",
                 fontsize=13, fontweight="bold")

    axes[0].plot(pct * 100, gain * 100, "#8e44ad", lw=2.5, label="Model")
    axes[0].plot([0, 100], [0, 100], "k--", alpha=0.4, label="Random")
    axes[0].fill_between(pct * 100, gain * 100, pct * 100,
                         alpha=0.1, color="#8e44ad")
    axes[0].set(xlabel="% Targeted", ylabel="% Positives Captured",
                title="Gain Chart")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(pct * 100, lift, "#e67e22", lw=2.5)
    axes[1].axhline(1, color="black", ls="--", alpha=0.4, label="Baseline")
    axes[1].set(xlabel="% Targeted", ylabel="Lift", title="Lift Chart")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, out / "fig15_lift.png")


def _fig16_calibration(y_test, ypr, out):
    fp, mp = calibration_curve(y_test, ypr, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Fig 16 | Calibration Curve — Final Model (M12)",
                 fontsize=13, fontweight="bold")
    ax.plot(mp, fp, "s-", color="#8e44ad", lw=2, label="Final Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set(xlabel="Mean Predicted Probability",
           ylabel="Fraction of Positives")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, out / "fig16_calibration.png")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
