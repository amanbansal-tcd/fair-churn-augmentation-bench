"""
eda.py
======
Exploratory Data Analysis — produces Figures 1–5.
"""
import logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import PALETTE, FIG_DIR, FIG_DPI, REPORT_DIR

logger   = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def run_eda(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Save Figures 1–5 to out_dir (default: FIG_DIR)."""
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    _fig1_target(df_raw, out)
    _fig2_numerical(df_clean, out)
    _fig3_categorical(df_clean, out)
    _fig4_chisq(df_clean, out)
    _fig5_outliers(df_clean, out)
    logger.info("EDA complete — 5 figures saved.")


# ── Figures ────────────────────────────────────────────────────────────────────

def _fig1_target(df, out):
    counts = df["Subscriber"].value_counts()
    pct    = counts / len(df) * 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fig 1 | Target Variable: Subscriber YES / NO",
                 fontsize=14, fontweight="bold")
    axes[0].bar(counts.index, counts.values,
                color=[PALETTE["YES"], PALETTE["NO"]],
                edgecolor="white", width=0.5)
    for i, (v, p) in enumerate(zip(counts.values, pct.values)):
        axes[0].text(i, v + 150, f"{v:,}\n({p:.1f}%)",
                     ha="center", fontweight="bold")
    axes[0].set(title="Class Distribution", ylabel="Count")
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=[PALETTE["YES"], PALETTE["NO"]], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proportion")
    axes[2].axis("off")
    txt = (f"CLASS IMBALANCE\n{'─'*30}\n"
           f"Total   : {len(df):,}\n"
           f"YES     : {counts['YES']:,} ({pct['YES']:.2f}%)\n"
           f"NO      : {counts['NO']:,} ({pct['NO']:.2f}%)\n"
           f"Ratio   : {counts['NO']/counts['YES']:.2f}:1")
    axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes, fontsize=11,
                 va="top", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#ecf0f1"))
    axes[2].set_title("Key Statistics")
    plt.tight_layout()
    _save(fig, out / "fig01_target.png")


def _fig2_numerical(df, out):
    feats = [("Year Of Residence", "Years Resident", 40),
             ("age_mid", "Age Midpoint", 14),
             ("fee_mid", "Weekly Fee ($)", 25),
             ("income_ord", "Income Level", 16)]
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle("Fig 2 | Numerical Distributions & Boxplots",
                 fontsize=14, fontweight="bold")
    for i, (feat, label, bins) in enumerate(feats):
        ax   = axes[0][i]
        data = df[feat].dropna()
        ax.hist(data, bins=bins, color=PALETTE["blue"],
                edgecolor="white", alpha=0.8)
        ax.axvline(data.mean(), color="red", ls="--", lw=1.8,
                   label=f"Mean {data.mean():.1f}")
        ax.axvline(data.median(), color="orange", ls="-", lw=1.8,
                   label=f"Med {data.median():.1f}")
        ax.set(title=label); ax.legend(fontsize=7)
    # Reward programme
    data = df["reward_prog_w"].dropna()
    axes[1][0].hist(data, bins=30, color=PALETTE["blue"],
                    edgecolor="white", alpha=0.8)
    axes[1][0].set(title="Reward Programme (winsorised)")
    for j in range(1, 4):
        axes[1][j].axis("off")
    # Boxplots by target
    for i, (feat, label, _) in enumerate(feats[:3]):
        ax  = axes[2][i]
        yes = df[df["target"] == 1][feat].dropna()
        no  = df[df["target"] == 0][feat].dropna()
        bp  = ax.boxplot([no, yes], tick_labels=["NO", "YES"],
                         patch_artist=True,
                         medianprops={"color": "black", "lw": 2})
        bp["boxes"][0].set_facecolor(PALETTE["NO"])
        bp["boxes"][1].set_facecolor(PALETTE["YES"])
        ax.set(title=f"{label} by Subscriber")
    # Correlation heatmap
    corr_cols = ["Year Of Residence", "age_mid", "fee_mid",
                 "income_ord", "reward_prog_w", "target"]
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=axes[2][3],
                cbar_kws={"shrink": 0.7}, square=True, linewidths=0.5)
    axes[2][3].set_title("Correlation Matrix")
    plt.tight_layout()
    _save(fig, out / "fig02_numerical.png")


def _fig3_categorical(df, out):
    overall = df["target"].mean()

    def bar_rate(ax, col, title, top=12, rot=45):
        grp = df.groupby(col)["target"].agg(["sum", "count"])
        grp["rate"] = grp["sum"] / grp["count"] * 100
        grp = grp.sort_values("count", ascending=False).head(top)
        colors = [
            PALETTE["YES"] if r >= overall * 100 else PALETTE["NO"]
            for r in grp["rate"]
        ]
        ax.bar(range(len(grp)), grp["rate"],
               color=colors, edgecolor="white", alpha=0.85)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp.index, rotation=rot, ha="right", fontsize=8)
        ax.axhline(overall * 100, color="black", ls="--", lw=1.5,
                   label=f"Avg {overall*100:.1f}%")
        for i, (v, n) in enumerate(zip(grp["rate"], grp["count"])):
            ax.text(i, v + 0.3, f"n={n}", ha="center", fontsize=6)
        ax.set(title=title, ylabel="Sub Rate (%)")
        ax.legend(fontsize=7)

    fig, axes = plt.subplots(3, 3, figsize=(22, 17))
    fig.suptitle("Fig 3 | Categorical Features vs Subscription Rate",
                 fontsize=14, fontweight="bold")
    bar_rate(axes[0][0], "HH Income",           "HH Income",          top=16, rot=60)
    bar_rate(axes[0][1], "Home Ownership",       "Home Ownership",     top=2,  rot=0)
    bar_rate(axes[0][2], "Ethnicity",            "Ethnicity (top 12)", top=12, rot=60)
    bar_rate(axes[1][0], "Language",             "Language (top 10)",  top=10, rot=45)
    bar_rate(axes[1][1], "Age range",            "Age Range",          top=13, rot=60)
    bar_rate(axes[1][2], "dummy for Children",   "Children in HH",     top=2,  rot=0)
    bar_rate(axes[2][0], "DP",                   "Delivery Period",    top=8,  rot=45)
    bar_rate(axes[2][1], "Source Channel",       "Source Channel",     top=12, rot=60)
    bar_rate(axes[2][2], "Nielsen Prizm",        "Nielsen PRIZM",      top=9,  rot=30)
    plt.tight_layout()
    _save(fig, out / "fig03_categorical.png")


def _fig4_chisq(df, out):
    cat_cols = [
        "HH Income", "Home Ownership", "Ethnicity", "Language",
        "Age range", "dummy for Children", "DP", "Source Channel", "Nielsen Prizm",
    ]
    rows = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col].fillna("Unknown"), df["target"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        cv = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
        rows.append({
            "Feature":  col,
            "Chi2":     round(chi2, 1),
            "p_value":  p,
            "CramersV": round(cv, 4),
        })
    chi_df = pd.DataFrame(rows).sort_values("CramersV", ascending=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    chi_df.to_csv(REPORT_DIR / "chi_square_results.csv", index=False)

    colors = [
        "#e74c3c" if p < 0.001 else "#f39c12" if p < 0.05 else "#95a5a6"
        for p in chi_df["p_value"]
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Fig 4 | Chi-Square Test — Cramér's V",
                 fontsize=13, fontweight="bold")
    ax.barh(range(len(chi_df)), chi_df["CramersV"],
            color=colors, edgecolor="white")
    ax.set_yticks(range(len(chi_df)))
    ax.set_yticklabels(chi_df["Feature"], fontsize=11)
    ax.set_xlabel("Cramér's V")
    for i, v in enumerate(chi_df["CramersV"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, out / "fig04_chisq.png")


def _fig5_outliers(df, out):
    feats = [
        ("Year Of Residence", "Years Resident"),
        ("age_mid",           "Age Midpoint"),
        ("fee_mid",           "Weekly Fee ($)"),
        ("reward_prog_w",     "Reward Programme"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Fig 5 | Outlier Analysis", fontsize=13, fontweight="bold")
    for ax, (feat, label) in zip(axes, feats):
        data = df[feat].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        n_out  = ((data < q1 - 1.5*(q3-q1)) | (data > q3 + 1.5*(q3-q1))).sum()
        bp     = ax.boxplot(data, patch_artist=True,
                            medianprops={"color": "black", "lw": 2})
        bp["boxes"][0].set_facecolor(PALETTE["blue"])
        ax.set_title(f"{label}\n{n_out} outliers ({n_out/len(data)*100:.1f}%)")
        ax.set_ylabel(label)
    plt.tight_layout()
    _save(fig, out / "fig05_outliers.png")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
