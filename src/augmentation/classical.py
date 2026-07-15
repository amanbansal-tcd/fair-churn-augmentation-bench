"""
classical.py
============
Classical data-augmentation techniques for minority-class oversampling,
plus the shared CSV-export, downstream-evaluation, and figure-plotting
utilities used by every augmentation technique in this package (including
CTGAN in ctgan_aug.py, and by extension HNSBA/LLM outputs once merged into
the same dataset registry by model_comparison.py).

Techniques implemented here
----------------------------
  1. SMOTE          - Synthetic Minority Over-sampling Technique
  2. ADASYN         - Adaptive Synthetic Sampling
  3. Gaussian Noise - Dual-mode noise injection (minority + majority)

Design principles
-----------------
- Augmentation is applied ONLY to the training set. The test set is never
  modified - it is a held-out reference copy.
- Each technique targets the minority class (Subscriber = YES).
- Gaussian Noise also expands the majority class (dual-mode operation).
- The 'source' column in all combined datasets marks real vs synthetic rows.

CSV output layout (outputs/augmented_datasets/)
------------------------------------------------
  train_original.csv          - raw 80% training split (no augmentation)
  test.csv                    - 20% held-out test set (never augmented)
  train_smote_combined.csv    - original + SMOTE synthetic rows
  train_smote_synthetic.csv   - SMOTE synthetic rows only
  train_adasyn_combined.csv   - original + ADASYN synthetic rows
  train_adasyn_synthetic.csv  - ADASYN synthetic rows only
  train_gaussian_combined.csv - original + Gaussian-perturbed rows
  train_gaussian_synthetic.csv- Gaussian synthetic rows only
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
import lightgbm as lgb

from src.config import (
    RANDOM_SEED, ALL_FEATURES, FEATURE_RANGES, FIG_DIR, FIG_DPI, AUG_DIR, REPORT_DIR,
    AUG_TARGET_MINORITY, AUG_SMOTE_K, AUG_ADASYN_K,
    AUG_GAUSS_NOISE_SCALE, AUG_GAUSS_MAJ_FRAC, LGB_PARAMS, AUG_COLORS,
)
from src.evaluation import evaluate_model

logger = logging.getLogger(__name__)

# Short display labels used in figure axis labels
FEAT_LABELS: dict = {
    "income_ord": "HH Income", "age_mid": "Age", "fee_mid": "Weekly Fee",
    "Year Of Residence": "Years Resid.", "reward_prog_w": "Reward Pts",
    "Home Ownership_enc": "HomeOwn", "Ethnicity_enc": "Ethnicity",
    "Language_enc": "Language", "dummy for Children_enc": "Children",
    "DP_enc": "Delivery", "Nielsen Prizm_enc": "PRIZM", "Source Channel_enc": "Channel",
}

# Numeric features to examine in distribution plots
NUM_FEATS: list = ["income_ord", "age_mid", "fee_mid", "Year Of Residence", "reward_prog_w"]

warnings.filterwarnings("ignore")


def run_augmentation(X_train: pd.DataFrame, y_train: pd.Series, run_ctgan: bool = True) -> dict:
    """
    Apply SMOTE, ADASYN, Gaussian Noise (and optionally CTGAN) to the
    training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for the training split only.
    y_train : pd.Series
        Binary target for the training split only (0 = NO, 1 = YES).
    run_ctgan : bool
        Set False to skip CTGAN (saves several minutes; useful for
        quick runs / CI). CTGAN itself lives in ctgan_aug.py and is
        imported lazily here to avoid a hard dependency for callers
        that never need it.

    Returns
    -------
    dict
        Keys: "original", "SMOTE", "ADASYN", "Gaussian Noise", "CTGAN".
        Each value: {"X": pd.DataFrame, "y": pd.Series}.
        "CTGAN" is omitted when run_ctgan=False or on backend failure.
    """
    # sampling_strategy must be a float in (0, 1]; it represents
    # target_minority / current_majority. Cap at 1.0 to avoid ValueError
    # when the requested target would exceed the majority class size.
    _n_majority = int((y_train == 0).sum())
    _target_min = min(AUG_TARGET_MINORITY, _n_majority)
    strat = float(_target_min) / float(_n_majority)

    # k_neighbors must be < n_minority_samples; clamp for safety on small folds.
    _n_minority = int((y_train == 1).sum())
    _smote_k = min(AUG_SMOTE_K, _n_minority - 1)
    _adasyn_k = min(AUG_ADASYN_K, _n_minority - 1)

    logger.info("Applying SMOTE (target minority=%d, k=%d)...", AUG_TARGET_MINORITY, _smote_k)
    sm = SMOTE(sampling_strategy=strat, k_neighbors=_smote_k, random_state=RANDOM_SEED)
    Xs, ys = sm.fit_resample(X_train, y_train)
    Xs = pd.DataFrame(Xs, columns=X_train.columns)
    ys = pd.Series(ys, name=y_train.name)
    logger.info("  SMOTE done: %d rows  (Class1=%d)", len(ys), (ys == 1).sum())

    logger.info("Applying ADASYN (target minority~%d, k=%d)...", AUG_TARGET_MINORITY, _adasyn_k)
    ad = ADASYN(sampling_strategy=strat, n_neighbors=_adasyn_k, random_state=RANDOM_SEED)
    Xa, ya = ad.fit_resample(X_train, y_train)
    Xa = pd.DataFrame(Xa, columns=X_train.columns)
    ya = pd.Series(ya, name=y_train.name)
    logger.info("  ADASYN done: %d rows  (Class1=%d)", len(ya), (ya == 1).sum())

    logger.info("Applying Gaussian Noise augmentation...")
    Xg, yg = gaussian_augment(
        X_train, y_train,
        target_minority=AUG_TARGET_MINORITY,
        majority_noise_frac=AUG_GAUSS_MAJ_FRAC,
        random_state=RANDOM_SEED,
    )
    logger.info("  Gaussian done: %d rows  (Class1=%d)", len(yg), (yg == 1).sum())

    aug = {
        "original": {"X": X_train, "y": y_train},
        "SMOTE": {"X": Xs, "y": ys},
        "ADASYN": {"X": Xa, "y": ya},
        "Gaussian Noise": {"X": Xg, "y": yg},
    }

    if run_ctgan:
        from src.config import AUG_CTGAN_EPOCHS, AUG_CTGAN_BATCH_SIZE, AUG_CTGAN_N_SYNTH
        from src.augmentation.ctgan_aug import ctgan_augment
        logger.info("Applying CTGAN (epochs=%d, n_synth=%d)...", AUG_CTGAN_EPOCHS, AUG_CTGAN_N_SYNTH)
        try:
            Xc, yc = ctgan_augment(
                X_train, y_train,
                n_synth=AUG_CTGAN_N_SYNTH,
                epochs=AUG_CTGAN_EPOCHS,
                batch_size=AUG_CTGAN_BATCH_SIZE,
                random_state=RANDOM_SEED,
            )
        except Exception as exc:
            logger.warning("CTGAN skipped due to backend error: %s", exc)
        else:
            aug["CTGAN"] = {"X": Xc, "y": yc}
            logger.info("  CTGAN done: %d rows  (Class1=%d)", len(yc), (yc == 1).sum())
    else:
        logger.info("CTGAN skipped (run_ctgan=False).")

    return aug


def gaussian_augment(
    X: pd.DataFrame,
    y: pd.Series,
    target_minority: int = 6_000,
    majority_noise_frac: float = 0.30,
    random_state: int = 42,
) -> tuple:
    """
    Dual-mode Gaussian Noise augmentation.

    Operation A - minority oversampling: resample minority records with
    replacement and add calibrated Gaussian noise (scale = 0.05 x
    per-feature std). Integer-encoded features are rounded; all values
    are clipped to the observed [min, max].

    Operation B - majority expansion: randomly select majority_noise_frac
    of majority records and perturb them with the same noise, growing the
    overall training set.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    target_minority : int
        Total minority-class count after augmentation.
    majority_noise_frac : float
        Fraction of majority rows to duplicate with noise (0 = disable).
    random_state : int

    Returns
    -------
    (X_aug, y_aug) : tuple[pd.DataFrame, pd.Series]
    """
    rng = np.random.default_rng(random_state)
    Xa = X.values.astype(float)
    ya = y.values
    fstd = Xa.std(axis=0)
    scale = AUG_GAUSS_NOISE_SCALE

    int_cols = [
        i for i, col in enumerate(X.columns)
        if col in FEATURE_RANGES and FEATURE_RANGES[col]["type"] == "int"
    ]

    Xmin = Xa[ya == 1]
    n_need = target_minority - len(Xmin)
    if n_need > 0:
        idx = rng.integers(0, len(Xmin), size=n_need)
        Xnew = Xmin[idx] + rng.normal(0, scale * fstd, (n_need, Xa.shape[1]))
        for j in range(Xa.shape[1]):
            Xnew[:, j] = np.clip(Xnew[:, j], Xa[:, j].min(), Xa[:, j].max())
        Xnew[:, int_cols] = np.round(Xnew[:, int_cols])
    else:
        Xnew = np.empty((0, Xa.shape[1]))

    Xmaj = Xa[ya == 0]
    n_exp = int(len(Xmaj) * majority_noise_frac)
    eidx = rng.choice(len(Xmaj), size=n_exp, replace=False)
    Xexp = Xmaj[eidx] + rng.normal(0, scale * fstd, (n_exp, Xa.shape[1]))
    for j in range(Xa.shape[1]):
        Xexp[:, j] = np.clip(Xexp[:, j], Xa[:, j].min(), Xa[:, j].max())
    Xexp[:, int_cols] = np.round(Xexp[:, int_cols])

    Xout = np.vstack([Xa, Xnew, Xexp])
    yout = np.concatenate([ya, np.ones(len(Xnew), dtype=int), np.zeros(n_exp, dtype=int)])
    perm = rng.permutation(len(Xout))
    return (
        pd.DataFrame(Xout[perm], columns=X.columns),
        pd.Series(yout[perm], name=y.name),
    )


def export_augmented_csvs(aug_dict: dict, X_test: pd.DataFrame, y_test: pd.Series, out_dir=None) -> None:
    """
    Write all augmented datasets to CSV files.

    Strategy for separating synthetic rows
    ----------------------------------------
    SMOTE and ADASYN append new rows after the original rows, so the tail
    slice beyond orig_len is guaranteed synthetic. Gaussian Noise (and
    CTGAN) shuffle the combined dataset, so positional slicing does not
    work; instead the synthetic-only portion is deterministically
    regenerated in isolation via _regen_synthetic (same random_state
    guarantees identical rows to what is embedded in the combined file).

    Parameters
    ----------
    aug_dict : dict
        Output of run_augmentation().
    X_test, y_test : pd.DataFrame, pd.Series
        Held-out test set (exported as reference; never modified).
    out_dir : str | Path | None
        Destination folder; defaults to AUG_DIR from config.
    """
    from pathlib import Path
    out = Path(out_dir) if out_dir else AUG_DIR
    out.mkdir(parents=True, exist_ok=True)

    orig_X = aug_dict["original"]["X"]
    orig_y = aug_dict["original"]["y"]
    n_orig = len(orig_X)

    _write_csv(orig_X, orig_y, out / "train_original.csv")
    logger.info("Saved train_original.csv  (%d rows)", n_orig)

    _write_csv(X_test, y_test, out / "test.csv")
    logger.info("Saved test.csv  (%d rows)", len(X_test))

    tech_slugs = {
        "SMOTE": "smote", "ADASYN": "adasyn",
        "Gaussian Noise": "gaussian", "CTGAN": "ctgan",
    }

    for tech, slug in tech_slugs.items():
        if tech not in aug_dict:
            continue
        X_aug = aug_dict[tech]["X"].reset_index(drop=True)
        y_aug = aug_dict[tech]["y"].reset_index(drop=True)
        n_aug = len(X_aug)
        n_new = n_aug - n_orig

        _write_csv(X_aug, y_aug, out / f"train_{slug}_combined.csv")
        logger.info("Saved train_%s_combined.csv  (%d rows, +%d synthetic)", slug, n_aug, max(0, n_new))

        if n_new > 0:
            if tech in ("SMOTE", "ADASYN"):
                synth_X = X_aug.iloc[n_orig:].reset_index(drop=True)
                synth_y = y_aug.iloc[n_orig:].reset_index(drop=True)
            else:
                synth_X, synth_y = _regen_synthetic(tech, orig_X, orig_y, n_new)
            _write_csv(synth_X, synth_y, out / f"train_{slug}_synthetic.csv")
            logger.info("Saved train_%s_synthetic.csv  (%d rows)", slug, len(synth_X))
        else:
            logger.warning("No synthetic rows for %s - synthetic file skipped.", tech)

    logger.info("All augmented CSVs written to: %s", out)
    _print_aug_summary(aug_dict)


def _regen_synthetic(tech: str, orig_X: pd.DataFrame, orig_y: pd.Series, n_new: int) -> tuple:
    """Regenerate synthetic-only rows for Gaussian Noise and CTGAN, deterministically."""
    if tech == "Gaussian Noise":
        rng = np.random.default_rng(RANDOM_SEED)
        Xa = orig_X.values.astype(float)
        ya = orig_y.values
        fstd = Xa.std(axis=0)
        scale = AUG_GAUSS_NOISE_SCALE
        int_cols = [
            i for i, col in enumerate(orig_X.columns)
            if col in FEATURE_RANGES and FEATURE_RANGES[col]["type"] == "int"
        ]

        Xmin = Xa[ya == 1]
        n_min = max(0, AUG_TARGET_MINORITY - len(Xmin))
        n_maj = int(len(Xa[ya == 0]) * AUG_GAUSS_MAJ_FRAC)

        if n_min > 0:
            idx_m = rng.integers(0, len(Xmin), size=n_min)
            Xnew = Xmin[idx_m] + rng.normal(0, scale * fstd, (n_min, Xa.shape[1]))
            for j in range(Xa.shape[1]):
                Xnew[:, j] = np.clip(Xnew[:, j], Xa[:, j].min(), Xa[:, j].max())
            Xnew[:, int_cols] = np.round(Xnew[:, int_cols])
        else:
            Xnew = np.empty((0, Xa.shape[1]))

        Xmaj = Xa[ya == 0]
        eidx = rng.choice(len(Xmaj), size=n_maj, replace=False)
        Xexp = Xmaj[eidx] + rng.normal(0, scale * fstd, (n_maj, Xa.shape[1]))
        for j in range(Xa.shape[1]):
            Xexp[:, j] = np.clip(Xexp[:, j], Xa[:, j].min(), Xa[:, j].max())
        Xexp[:, int_cols] = np.round(Xexp[:, int_cols])

        X_synth = pd.DataFrame(np.vstack([Xnew, Xexp]), columns=orig_X.columns)
        y_synth = pd.Series(
            np.concatenate([np.ones(len(Xnew), dtype=int), np.zeros(n_maj, dtype=int)]),
            name=orig_y.name,
        )
    elif tech == "CTGAN":
        from src.augmentation.ctgan_aug import regen_ctgan_synthetic
        X_synth, y_synth = regen_ctgan_synthetic(orig_X, orig_y, n_new)
    else:
        raise ValueError(f"Unknown tech for regeneration: {tech}")

    return X_synth.reset_index(drop=True), y_synth.reset_index(drop=True)


def _write_csv(X: pd.DataFrame, y: pd.Series, path) -> None:
    """Write features + target to a CSV file."""
    df = X.copy().reset_index(drop=True)
    df["target"] = y.values
    df.to_csv(path, index=False)


def _print_aug_summary(aug_dict: dict) -> None:
    """Print a formatted summary table of all augmented datasets."""
    orig_size = len(aug_dict["original"]["y"])
    print("\n" + "=" * 75)
    print("  AUGMENTED DATASET SUMMARY")
    print("=" * 75)
    print(f"  {'Technique':<22} {'Total':>8} {'Class 0':>8} {'Class 1':>8} {'Ratio':>8} {'New rows':>10}")
    print("-" * 75)
    for name, dset in aug_dict.items():
        y = dset["y"]
        tot = len(y)
        c0 = int((y == 0).sum())
        c1 = int((y == 1).sum())
        rt = c0 / c1 if c1 > 0 else 0.0
        new = tot - orig_size
        tag = f"+{new:,}" if new > 0 else "-"
        print(f"  {name:<22} {tot:>8,} {c0:>8,} {c1:>8,} {rt:>7.2f}:1 {tag:>10}")
    print("=" * 75 + "\n")


def run_augmentation_evaluation(aug_dict: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Train an identical LightGBM on each augmented dataset and evaluate
    against the unchanged held-out test set. No class_weight is used:
    the augmented datasets already handle imbalance via oversampling.

    Returns
    -------
    dict
        {technique_name: metrics_dict}; metrics_dict includes all keys
        from evaluate_model() plus "model".
    """
    results = {}
    for name, dset in aug_dict.items():
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(dset["X"], dset["y"])
        yp = m.predict(X_test)
        ypr = m.predict_proba(X_test)[:, 1]
        results[name] = {**evaluate_model(y_test, yp, ypr), "model": m}
        logger.info(
            "%-22s  AUC=%.4f  F1=%.4f  Recall=%.4f  BalAcc=%.4f",
            name, results[name]["roc_auc"], results[name]["f1"],
            results[name]["recall"], results[name]["bal_acc"],
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "Technique": n,
            "ROC_AUC": round(r["roc_auc"], 4),
            "F1": round(r["f1"], 4),
            "Recall": round(r["recall"], 4),
            "Precision": round(r["precision"], 4),
            "Bal_Acc": round(r["bal_acc"], 4),
            "MCC": round(r["mcc"], 4),
        }
        for n, r in results.items()
    ]).to_csv(REPORT_DIR / "augmentation_results.csv", index=False)
    logger.info("Augmentation results CSV saved.")

    return results


# Figures A-I: augmentation comparison plots

def plot_augmentation_figures(aug_dict: dict, model_results: dict, y_test: pd.Series, out_dir=None) -> None:
    """Save all nine augmentation comparison figures (A-I)."""
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    _figA_class_dist(aug_dict, out)
    _figB_feature_dist(aug_dict, out)
    _figC_ks_fidelity(aug_dict, out)
    _figD_pca_scatter(aug_dict, out)
    _figE_correlation(aug_dict, out)
    _figF_performance(model_results, out)
    _figG_roc_pr(model_results, y_test, out)
    _figH_stats(aug_dict, out)
    _figI_radar(model_results, out)

    logger.info("Augmentation figures A-I saved to %s", out)


def _get_color(name: str) -> str:
    """Return AUG_COLORS entry, falling back to grey for unknown names."""
    return AUG_COLORS.get(name, AUG_COLORS.get(name.lower(), "#95a5a6"))


def _figA_class_dist(aug: dict, out: Path) -> None:
    names = list(aug.keys())
    sizes = [len(d["y"]) for d in aug.values()]
    c0s = [int((d["y"] == 0).sum()) for d in aug.values()]
    c1s = [int((d["y"] == 1).sum()) for d in aug.values()]
    ratios = [c0 / c1 for c0, c1 in zip(c0s, c1s)]
    orig = sizes[0]
    colors = [_get_color(n) for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Fig A | Class Distribution After Augmentation", fontsize=14, fontweight="bold")

    axes[0].bar(x, c0s, color="#e74c3c", alpha=0.85, label="Class 0 (NO)")
    axes[0].bar(x, c1s, bottom=c0s, color="#27ae60", alpha=0.85, label="Class 1 (YES)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_title("Sample Counts")
    axes[0].legend()
    for i, (t, c0, c1) in enumerate(zip(sizes, c0s, c1s)):
        axes[0].text(i, t + 200, f"{t:,}", ha="center", fontsize=8, fontweight="bold")

    axes[1].bar(names, ratios, color=colors, edgecolor="white", alpha=0.9)
    axes[1].axhline(1.0, color="black", ls="--", lw=1.5, label="Balance (1:1)")
    axes[1].set_title("Imbalance Ratio")
    axes[1].legend(fontsize=8)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    for i, r in enumerate(ratios):
        axes[1].text(i, r + 0.06, f"{r:.2f}:1", ha="center", fontsize=9, fontweight="bold")

    axes[2].bar(names, sizes, color=colors, edgecolor="white", alpha=0.9)
    axes[2].axhline(orig, color="black", ls="--", lw=1.5, label=f"Original ({orig:,})")
    axes[2].set_title("Dataset Size")
    axes[2].legend(fontsize=8)
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    for i, s in enumerate(sizes):
        g = s - orig
        lbl = f"{s:,}\n(+{g:,})" if g > 0 else f"{s:,}"
        axes[2].text(i, s + 150, lbl, ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    _save(fig, out / "figA_class_dist.png")


def _figB_feature_dist(aug: dict, out: Path) -> None:
    n_techs = len(aug)
    fig, axes = plt.subplots(len(NUM_FEATS), n_techs, figsize=(5 * n_techs, 18))
    fig.suptitle("Fig B | Minority Class Feature Distributions", fontsize=13, fontweight="bold")

    for row, feat in enumerate(NUM_FEATS):
        for col, (name, dset) in enumerate(aug.items()):
            ax = axes[row][col]
            mask = (dset["y"] == 1)
            data = dset["X"].loc[mask, feat].values
            ax.hist(data, bins=30, color=_get_color(name), edgecolor="white", alpha=0.8, density=True)
            try:
                kde = stats.gaussian_kde(data)
                kx = np.linspace(data.min(), data.max(), 200)
                ax.plot(kx, kde(kx), "k-", lw=1.5)
            except Exception:
                pass
            ax.set_title(f"{name}\n{FEAT_LABELS.get(feat, feat)}  n={len(data):,}", fontsize=8, fontweight="bold")
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save(fig, out / "figB_feature_dist.png")


def _figC_ks_fidelity(aug: dict, out: Path) -> None:
    orig_X = aug["original"]["X"]
    orig_min = orig_X[aug["original"]["y"] == 1]
    techs = [t for t in aug if t != "original"]

    ks_rows = {}
    for tech in techs:
        dset = aug[tech]
        synth_min = dset["X"][dset["y"] == 1]
        ks_rows[tech] = {
            FEAT_LABELS.get(f, f): round(ks_2samp(orig_min[f].values, synth_min[f].values)[0], 4)
            for f in ALL_FEATURES if f in orig_min.columns
        }
    ks_df = pd.DataFrame(ks_rows).T

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle("Fig C | KS-Test Fidelity (lower = better)", fontsize=13, fontweight="bold")
    sns.heatmap(ks_df, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=axes[0], linewidths=0.5,
                cbar_kws={"label": "KS Statistic"})
    axes[0].set_title("KS Statistic per Feature per Technique")

    mean_ks = ks_df.mean(axis=1)
    axes[1].bar(mean_ks.index, mean_ks.values, color=[_get_color(t) for t in mean_ks.index],
                edgecolor="white", alpha=0.9)
    axes[1].set_title("Mean KS (all features)")
    for i, (t, v) in enumerate(mean_ks.items()):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticklabels(mean_ks.index, rotation=20, ha="right")

    plt.tight_layout()
    _save(fig, out / "figC_ks_fidelity.png")


def _figD_pca_scatter(aug: dict, out: Path) -> None:
    orig_X = aug["original"]["X"]
    sc = StandardScaler()
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(sc.fit_transform(orig_X))
    rng = np.random.default_rng(RANDOM_SEED)
    techs = [t for t in aug if t != "original"]

    fig, axes = plt.subplots(1, len(techs), figsize=(7 * len(techs), 7))
    if len(techs) == 1:
        axes = [axes]
    fig.suptitle("Fig D | PCA 2D - Real vs Synthetic Minority Samples", fontsize=13, fontweight="bold")

    for ax, tech in zip(axes, techs):
        dset = aug[tech]
        X_aug = dset["X"]
        y_aug = np.asarray(dset["y"])
        is_real = X_aug.index.isin(orig_X.index) if hasattr(X_aug, "index") else np.zeros(len(X_aug), bool)
        is_min = y_aug == 1
        is_maj = y_aug == 0
        Xp = pca.transform(sc.transform(X_aug))

        maj_i = np.where(is_maj)[0]
        s_maj = rng.choice(maj_i, size=min(500, len(maj_i)), replace=False)
        ax.scatter(Xp[s_maj, 0], Xp[s_maj, 1], c="#bdc3c7", alpha=0.3, s=8, zorder=1, label="Majority")

        real_i = np.where(is_min & is_real)[0]
        if len(real_i):
            s_r = rng.choice(real_i, size=min(300, len(real_i)), replace=False)
            ax.scatter(Xp[s_r, 0], Xp[s_r, 1], c="#2980b9", alpha=0.7, s=25, zorder=3, label="Real minority")

        syn_i = np.where(is_min & ~is_real)[0]
        if len(syn_i):
            s_s = rng.choice(syn_i, size=min(300, len(syn_i)), replace=False)
            ax.scatter(Xp[s_s, 0], Xp[s_s, 1], c="#e67e22", alpha=0.6, s=25, zorder=2, marker="^",
                       label=f"Synthetic ({len(syn_i):,})")

        ev = pca.explained_variance_ratio_
        ax.set_title(f"{tech}\n{len(syn_i):,} synthetic samples", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, out / "figD_pca_scatter.png")


def _figE_correlation(aug: dict, out: Path) -> None:
    fl = [FEAT_LABELS.get(f, f) for f in ALL_FEATURES]
    n = len(aug)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle("Fig E | Correlation Matrix - Minority Class", fontsize=13, fontweight="bold")

    for ax, (name, dset) in zip(axes, aug.items()):
        Xm = dset["X"][dset["y"] == 1].copy()
        Xm.columns = fl
        mask = np.triu(np.ones(len(fl), bool))
        sns.heatmap(Xm.corr(), mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                    square=True, cbar=False, annot_kws={"size": max(4, 8 - n)})
        ax.set_title(f"{name}\n(n={len(Xm):,})", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save(fig, out / "figE_correlation.png")


def _figF_performance(model_results: dict, out: Path) -> None:
    names = list(model_results.keys())
    metrics = [("roc_auc", "ROC-AUC"), ("f1", "F1 Score"), ("bal_acc", "Balanced Accuracy"), ("recall", "Recall")]
    orig_res = model_results.get("original", list(model_results.values())[0])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Fig F | Model Performance per Augmentation Technique\n(evaluated on unchanged test set)",
                 fontsize=13, fontweight="bold")

    for ax, (mk, ml) in zip(axes.flat, metrics):
        vals = [model_results[n][mk] for n in names]
        colors = [_get_color(n) for n in names]
        ax.bar(range(len(names)), vals, color=colors, edgecolor="white", alpha=0.9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(ml, fontweight="bold")
        ax.set_ylim(min(vals) * 0.93, max(vals) * 1.06)
        ax.grid(alpha=0.3, axis="y")

        orig_v = orig_res[mk]
        for i, (n, v) in enumerate(zip(names, vals)):
            d = v - orig_v
            sign = "+" if d >= 0 else ""
            col = "#27ae60" if d > 0 else "#e74c3c" if d < 0 else "grey"
            ax.text(i, v + (max(vals) - min(vals)) * 0.01, f"{v:.4f}\n({sign}{d:.4f})",
                    ha="center", fontsize=8, fontweight="bold", color=col)

    plt.tight_layout()
    _save(fig, out / "figF_aug_performance.png")


def _figG_roc_pr(model_results: dict, y_test: pd.Series, out: Path) -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Fig G | ROC & Precision-Recall Curves - Augmentation Comparison", fontsize=13, fontweight="bold")

    for name, res in model_results.items():
        color = _get_color(name)
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, color=color, lw=2.5, label=f"{name} AUC={res['roc_auc']:.4f}")
        p, r, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(r, p, color=color, lw=2.5, label=f"{name} PR={res['pr_auc']:.4f}")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(np.asarray(y_test).mean(), color="black", ls="--", alpha=0.5, label="Chance")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR Curves")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, out / "figG_aug_roc_pr.png")


def _figH_stats(aug: dict, out: Path) -> None:
    names = list(aug.keys())
    fig, axes = plt.subplots(1, len(NUM_FEATS), figsize=(22, 6))
    fig.suptitle("Fig H | Minority Class: Mean +/- Std per Feature", fontsize=12, fontweight="bold")

    for ax, feat in zip(axes, NUM_FEATS):
        means, stds = [], []
        for dset in aug.values():
            d = dset["X"].loc[dset["y"] == 1, feat]
            means.append(d.mean())
            stds.append(d.std())
        ax.bar(range(len(names)), means, yerr=stds, color=[_get_color(n) for n in names],
               edgecolor="white", alpha=0.85, capsize=5, error_kw={"elinewidth": 2})
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        ax.set_title(FEAT_LABELS.get(feat, feat), fontweight="bold")
        ax.axhline(means[0], color="black", ls="--", lw=1.2, alpha=0.6)
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, out / "figH_aug_stats.png")


def _figI_radar(model_results: dict, out: Path) -> None:
    metric_keys = ["roc_auc", "f1", "bal_acc", "recall", "precision", "mcc"]
    labels = ["ROC-AUC", "F1", "Bal.Acc", "Recall", "Precision", "MCC"]
    names = list(model_results.keys())
    arr = np.array([[model_results[n][m] for m in metric_keys] for n in names])
    mn, mx = arr.min(0), arr.max(0)
    norm = (arr - mn) / (mx - mn + 1e-9)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle("Fig I | Holistic Performance Radar\n(normalised - outer edge = best per metric)",
                 fontsize=13, fontweight="bold", y=1.02)

    for name, row in zip(names, norm):
        vals = row.tolist() + [row[0]]
        color = _get_color(name)
        ax.plot(angles, vals, "o-", lw=2.5, color=color, label=name, markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="grey")
    ax.grid(color="grey", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.15), fontsize=10)

    plt.tight_layout()
    _save(fig, out / "figI_aug_radar.png")


def _save(fig, path: Path) -> None:
    """Save and close a matplotlib figure."""
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
