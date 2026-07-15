"""
pareto.py
=========
Pareto frontier analysis and master model comparison.

Figures produced
----------------
  Fig 11 — Fairness–performance Pareto scatter
  Fig 12 — Model comparison heatmap
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIG_DIR, FIG_DPI, REPORT_DIR

logger = logging.getLogger(__name__)

TYPE_COLORS  = {
    "Baseline":    "#e74c3c",
    "Bias-Removal":"#3498db",
    "Rebalancing": "#27ae60",
}
TYPE_MARKERS = {
    "Baseline":    "*",
    "Bias-Removal":"o",
    "Rebalancing": "s",
}


def build_master_table(
    baseline_results: dict,
    mitigation_df: pd.DataFrame,
    rebalancing_df: pd.DataFrame,
    baseline_fairness: dict | None = None,
) -> pd.DataFrame:
    """
    Assemble all experiments into one master comparison table.

    Parameters
    ----------
    baseline_results : dict
        Output of train_all().
    mitigation_df : pd.DataFrame
        Output of run_mitigation_experiments().
    rebalancing_df : pd.DataFrame
        Output of run_rebalancing_experiments().
    baseline_fairness : dict | None
        Optional fairness audit on the CatBoost baseline.

    Returns
    -------
    pd.DataFrame
        Saved to outputs/reports/master_comparison.csv.
    """
    rows = []

    # Baseline models
    for name, res in baseline_results.items():
        dpd = dir_ = eod = None
        if baseline_fairness and name == "CatBoost":
            m = baseline_fairness.get("Ethnicity", {}).get("metrics", {})
            dpd, dir_, eod = m.get("DPD"), m.get("DIR"), m.get("EOD")
        rows.append({
            "Model": name, "Type": "Baseline",
            "Accuracy":  round(res["accuracy"],  4),
            "Precision": round(res["precision"], 4),
            "Recall":    round(res["recall"],    4),
            "F1":        round(res["f1"],        4),
            "ROC_AUC":   round(res["roc_auc"],   4),
            "Bal_Acc":   round(res["bal_acc"],   4),
            "Log_Loss":  round(res["log_loss"],  4),
            "MCC":       round(res["mcc"],       4),
            "DPD":       dpd, "DIR": dir_, "EOD": eod,
        })

    # Bias-removal variants
    for _, row in mitigation_df.iterrows():
        rows.append({
            "Model": row["Model"], "Type": "Bias-Removal",
            **{k: row[k] for k in [
                "Accuracy","Precision","Recall","F1",
                "ROC_AUC","Bal_Acc","Log_Loss","MCC","DPD","DIR","EOD",
            ]},
        })

    # Rebalancing variants
    for _, row in rebalancing_df.iterrows():
        rows.append({
            "Model": row["Model"], "Type": "Rebalancing",
            **{k: row[k] for k in [
                "Accuracy","Precision","Recall","F1",
                "ROC_AUC","Bal_Acc","Log_Loss","MCC","DPD","DIR","EOD",
            ]},
        })

    master = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(REPORT_DIR / "master_comparison.csv", index=False)
    logger.info("Master comparison saved (%d models).", len(master))
    return master


def pareto_frontier(
    master: pd.DataFrame,
    perf_col: str = "ROC_AUC",
    fair_col: str = "DPD",
) -> pd.DataFrame:
    """
    Return the non-dominated subset on the perf_col vs fair_col plane.
    A model dominates another if it has higher perf AND lower fair.
    """
    mf   = master.dropna(subset=[perf_col, fair_col]).copy()
    pts  = mf[[fair_col, perf_col]].values
    keep = [
        i for i, p in enumerate(pts)
        if not any(
            j != i
            and pts[j, 0] <= p[0]   # lower (fairer) DPD
            and pts[j, 1] >= p[1]   # higher perf
            and (pts[j, 0] < p[0] or pts[j, 1] > p[1])
            for j in range(len(pts))
        )
    ]
    return mf.iloc[keep]


def plot_pareto(
    master: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Save Fig 11: Pareto frontier scatter for AUC and F1 vs DPD."""
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    mf  = master.dropna(subset=["DPD", "ROC_AUC"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Fig 11 | Fairness–Performance Pareto Frontier",
                 fontsize=14, fontweight="bold")

    for ax, (yc, yl) in zip(
        axes, [("ROC_AUC", "ROC-AUC"), ("F1", "F1 Score")]
    ):
        for t, sub in mf.groupby("Type"):
            ax.scatter(
                sub["DPD"], sub[yc],
                c=TYPE_COLORS.get(t, "grey"),
                marker=TYPE_MARKERS.get(t, "o"),
                s=120, label=t,
                edgecolors="white", linewidths=0.8, zorder=3,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    row["Model"], (row["DPD"], row[yc]),
                    textcoords="offset points",
                    xytext=(4, 4), fontsize=7, alpha=0.8,
                )
        pf = pareto_frontier(mf, yc, "DPD").sort_values("DPD")
        ax.plot(pf["DPD"], pf[yc], "k--", lw=1.5,
                alpha=0.5, label="Pareto frontier")
        ax.set(
            xlabel="Demographic Parity Diff (lower = fairer)",
            ylabel=yl,
            title=f"{yl} vs Fairness",
        )
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, out / "fig11_pareto.png")


def plot_heatmap(
    master: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Save Fig 12: normalised model comparison heatmap."""
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    mf  = master.dropna(subset=["DPD", "DIR"]).copy()

    pdf = mf[["Model","ROC_AUC","F1","Bal_Acc","DPD","DIR"]].set_index("Model")
    pdf.columns = ["ROC-AUC","F1","Bal.Acc","DPD↓","DIR↑"]

    norm = pdf.copy().astype(float)
    for c in ["ROC-AUC","F1","Bal.Acc","DIR↑"]:
        r        = norm[c].max() - norm[c].min()
        norm[c]  = (norm[c] - norm[c].min()) / (r + 1e-9)
    r              = norm["DPD↓"].max() - norm["DPD↓"].min()
    norm["DPD↓"]   = 1 - (norm["DPD↓"] - norm["DPD↓"].min()) / (r + 1e-9)

    fig, ax = plt.subplots(figsize=(14, max(8, len(mf) // 2)))
    fig.suptitle("Fig 12 | Model Comparison Heatmap",
                 fontsize=14, fontweight="bold")
    sns.heatmap(
        norm, annot=pdf.round(4), fmt="",
        cmap="RdYlGn", linewidths=0.5, ax=ax,
        cbar_kws={"label": "Normalised Score (green = best)"},
    )
    plt.tight_layout()
    _save(fig, out / "fig12_heatmap.png")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
