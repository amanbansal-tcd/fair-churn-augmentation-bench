"""
fairness.py
===========
Group fairness metrics and bias audit dashboard.

Eight metrics are computed per sensitive attribute:
  DPD, DPR, DIR, EOD, EqOD, FPR_D, FNR_D, SPD

Figure produced
---------------
  Fig 10 — Fairness audit dashboard (tables + bar charts)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import (
    SENSITIVE_ATTRS, FAIRNESS_THRESHOLDS,
    FIG_DIR, FIG_DPI, REPORT_DIR,
)

logger  = logging.getLogger(__name__)
MIN_N   = FAIRNESS_THRESHOLDS["MIN_GROUP_N"]


# ── Public API ─────────────────────────────────────────────────────────────────

def run_fairness_audit(
    y_true,
    y_pred,
    y_prob,
    df: pd.DataFrame,
    test_idx,
    attrs: list[str] | None = None,
) -> dict:
    """
    Compute fairness metrics for every specified sensitive attribute.

    Parameters
    ----------
    y_true, y_pred, y_prob : array-like
        Ground truth, hard predictions, and probabilities for the test set.
    df : pd.DataFrame
        Full (cleaned) DataFrame that contains the sensitive columns.
    test_idx : pd.Index
        Index values of the test set rows (used to look up sensitive cols).
    attrs : list[str] | None
        Attribute names to audit; defaults to SENSITIVE_ATTRS from config.

    Returns
    -------
    dict
        {attr_name: {"groups": pd.DataFrame, "metrics": dict}}
    """
    attrs  = attrs or SENSITIVE_ATTRS
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    audit  = {}

    for attr in attrs:
        if attr not in df.columns:
            logger.warning("Attribute '%s' not in df — skipping.", attr)
            continue
        gdf, met = _group_metrics(y_true, y_pred, y_prob, df, test_idx, attr)
        audit[attr] = {"groups": gdf, "metrics": met}
        logger.info(
            "Fairness [%-18s]:  DPD=%.4f  DIR=%.4f",
            attr, met.get("DPD", 0), met.get("DIR", 0),
        )

    return audit


def fairness_summary_df(audit: dict) -> pd.DataFrame:
    """Flatten audit dict to a single-row-per-attribute DataFrame."""
    return pd.DataFrame([
        {"Attribute": a, **v["metrics"]}
        for a, v in audit.items()
    ])


def plot_fairness_dashboard(
    audit: dict,
    out_dir: str | Path | None = None,
) -> None:
    """
    Save Fig 10: fairness dashboard with metric table and per-group bar charts.
    Also exports fairness_summary.csv to REPORT_DIR.
    """
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(26, 22))
    fig.suptitle("Fig 10 | Fairness & Bias Audit Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    gs  = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38)

    # ── Metric summary table ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    met_keys   = ["DPD", "DPR", "DIR", "EOD", "FPR_D", "FNR_D", "SPD"]
    met_labels = [
        "Dem.Parity Diff", "Dem.Parity Ratio", "Dis.Impact",
        "Equal Opp Diff",  "FPR Diff", "FNR Diff", "Stat.Parity Diff",
    ]
    tdata = [
        [a] + [f"{v['metrics'].get(m, 0):.4f}" for m in met_keys]
        for a, v in audit.items()
    ]
    tbl = ax0.table(
        cellText=tdata,
        colLabels=["Attribute"] + met_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and c > 0:
            try:
                val = float(cell.get_text().get_text())
                # Red = severe, Orange = moderate, Green = acceptable
                if c in [2, 3]:   # ratio-type metrics (lower is worse)
                    fc = ("#e74c3c" if val < 0.60
                          else "#f39c12" if val < 0.80
                          else "#2ecc71")
                else:              # difference-type metrics (higher is worse)
                    fc = ("#e74c3c" if val > 0.20
                          else "#f39c12" if val > 0.10
                          else "#2ecc71")
                cell.set_facecolor(fc)
                cell.set_text_props(color="white" if fc != "#2ecc71" else "black")
            except Exception:
                pass
    ax0.set_title("Red=Severe | Orange=Moderate | Green=Acceptable",
                  fontsize=10, fontweight="bold", pad=25)

    # ── Per-group predicted positive rates ────────────────────────────────────
    overall = 0.191
    for ci, attr in enumerate(["Ethnicity", "Language", "Home Ownership"]):
        if attr not in audit:
            continue
        ax  = fig.add_subplot(gs[1, ci])
        gdf = audit[attr]["groups"]
        colors = [
            "#e74c3c" if v < overall * 0.7 else
            "#f39c12" if v < overall else "#3498db"
            for v in gdf["Pred_Pos_Rate"]
        ]
        ax.barh(range(len(gdf)), gdf["Pred_Pos_Rate"] * 100,
                color=colors, edgecolor="white")
        ax.set_yticks(range(len(gdf)))
        ax.set_yticklabels(gdf["Group"].astype(str), fontsize=8)
        ax.axvline(overall * 100, color="black", ls="--", lw=1.5)
        ax.set(xlabel="Predicted Sub Rate (%)", title=f"{attr}")

    # ── TPR disparity ─────────────────────────────────────────────────────────
    for ci, attr in enumerate(["Ethnicity", "Language", "Age range"]):
        if attr not in audit:
            continue
        ax  = fig.add_subplot(gs[2, ci])
        gdf = audit[attr]["groups"]
        colors = [
            "#e74c3c" if v < 0.55 else
            "#f39c12" if v < 0.70 else "#2ecc71"
            for v in gdf["TPR"]
        ]
        ax.barh(range(len(gdf)), gdf["TPR"] * 100,
                color=colors, edgecolor="white")
        ax.set_yticks(range(len(gdf)))
        ax.set_yticklabels(gdf["Group"].astype(str), fontsize=8)
        ax.axvline(70, color="black", ls="--", lw=1.5)
        ax.set(xlabel="TPR / Recall (%)", title=f"{attr} — TPR Disparity")
        ax.set_xlim(0, 110)

    fig.savefig(out / "fig10_fairness.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig10_fairness.png")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fairness_summary_df(audit).to_csv(
        REPORT_DIR / "fairness_summary.csv", index=False
    )


# ── Internal helper ────────────────────────────────────────────────────────────

def _group_metrics(y_true, y_pred, y_prob, df, test_idx, attr):
    """
    Compute per-group statistics and aggregate fairness metrics.

    Groups smaller than MIN_GROUP_N are excluded.
    """
    sensitive = df.loc[test_idx, attr].fillna("Unknown").values
    rows = []

    for g in np.unique(sensitive):
        m = sensitive == g
        if m.sum() < MIN_N:
            continue
        yt, yp = y_true[m], y_pred[m]
        ppr = yp.mean()
        ar  = yt.mean()

        if len(np.unique(yt)) > 1:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        else:
            tn = (yp == 0).sum(); fp = fn = 0; tp = (yp == 1).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        rows.append(dict(
            Group=g, N=m.sum(),
            Actual_Rate=round(ar, 4),
            Pred_Pos_Rate=round(ppr, 4),
            TPR=round(tpr, 4),
            FPR=round(fpr, 4),
            FNR=round(fnr, 4),
        ))

    gdf = pd.DataFrame(rows)
    if gdf.empty:
        return gdf, {}

    mx_ppr, mn_ppr = gdf["Pred_Pos_Rate"].max(), gdf["Pred_Pos_Rate"].min()
    mx_tpr, mn_tpr = gdf["TPR"].max(), gdf["TPR"].min()
    mx_fpr, mn_fpr = gdf["FPR"].max(), gdf["FPR"].min()
    mx_fnr, mn_fnr = gdf["FNR"].max(), gdf["FNR"].min()

    met = {
        "DPD":   round(mx_ppr - mn_ppr, 4),
        "DPR":   round(mn_ppr / mx_ppr, 4) if mx_ppr > 0 else 1.0,
        "DIR":   round(mn_ppr / mx_ppr, 4) if mx_ppr > 0 else 1.0,
        "EOD":   round(mx_tpr - mn_tpr, 4),
        "EqOD":  round(max(mx_tpr - mn_tpr, mx_fpr - mn_fpr), 4),
        "FPR_D": round(mx_fpr - mn_fpr, 4),
        "FNR_D": round(mx_fnr - mn_fnr, 4),
        "SPD":   round(mx_ppr - mn_ppr, 4),
    }
    return gdf, met
