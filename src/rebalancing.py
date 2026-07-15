"""
rebalancing.py
==============
R1–R6 fairness rebalancing experiments.

Techniques
----------
  R1 — Class weight balancing (class_weight='balanced')
  R2 — SMOTE oversampling
  R3 — Ethnicity-proportional sample weights
  R4 — Language-proportional sample weights
  R5 — Home Ownership-proportional sample weights
  R6 — Combined Eth × Lang × HO reweighting
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

from src.config import ALL_FEATURES, LGB_PARAMS, RANDOM_SEED, REPORT_DIR
from src.evaluation import evaluate_model
from src.fairness import _group_metrics

logger = logging.getLogger(__name__)


def run_rebalancing_experiments(
    df: pd.DataFrame,
    X_train, X_test,
    y_train, y_test,
    fairness_attr: str = "Ethnicity",
) -> pd.DataFrame:
    """
    Train R1–R6 and return a results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame (for group-weight computation).
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    fairness_attr : str

    Returns
    -------
    pd.DataFrame
    """
    feats = [f for f in ALL_FEATURES if f in X_train.columns]
    rows  = []

    # ── R1: class_weight='balanced' ───────────────────────────────────────────
    rows.append(_run(
        "R1_ClassWeight", "Class weight balancing",
        X_train[feats], y_train, X_test[feats], y_test,
        df, X_test.index, fairness_attr,
        use_balance=True,
    ))

    # ── R2: SMOTE oversampling ────────────────────────────────────────────────
    sm         = SMOTE(random_state=RANDOM_SEED)
    Xsm, ysm   = sm.fit_resample(X_train[feats], y_train)
    rows.append(_run(
        "R2_SMOTE", "SMOTE oversampling",
        pd.DataFrame(Xsm, columns=feats), pd.Series(ysm),
        X_test[feats], y_test,
        df, X_test.index, fairness_attr,
    ))

    # ── R3–R5: group reweighting ──────────────────────────────────────────────
    for mid, desc, col in [
        ("R3_EthReweight",  "Ethnicity reweighting",    "Ethnicity"),
        ("R4_LangReweight", "Language reweighting",     "Language"),
        ("R5_HOwnReweight", "Home Ownership reweighting","Home Ownership"),
    ]:
        w = _group_weights(df, X_train.index, col)
        rows.append(_run(
            mid, desc,
            X_train[feats], y_train, X_test[feats], y_test,
            df, X_test.index, fairness_attr,
            weights=w,
        ))

    # ── R6: combined reweighting ──────────────────────────────────────────────
    we  = _group_weights(df, X_train.index, "Ethnicity")
    wl  = _group_weights(df, X_train.index, "Language")
    wh  = _group_weights(df, X_train.index, "Home Ownership")
    wc  = we * wl * wh
    wc  = wc / wc.mean()          # normalise to unit mean
    rows.append(_run(
        "R6_Combined", "Eth × Lang × HO combined reweighting",
        X_train[feats], y_train, X_test[feats], y_test,
        df, X_test.index, fairness_attr,
        weights=wc,
    ))

    df_out = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(REPORT_DIR / "rebalancing_results.csv", index=False)
    logger.info("Rebalancing results saved (%d variants).", len(df_out))
    return df_out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(
    mid: str, desc: str,
    Xtr, ytr, Xte, yte,
    df, test_idx, attr,
    weights=None,
    use_balance: bool = False,
) -> dict:
    params = dict(LGB_PARAMS)
    if use_balance:
        params["class_weight"] = "balanced"
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, sample_weight=weights)

    yp   = m.predict(Xte)
    ypr  = m.predict_proba(Xte)[:, 1]
    met  = evaluate_model(yte, yp, ypr)
    _, fmet = _group_metrics(yte.values, yp, ypr, df, test_idx, attr)

    logger.info(
        "%s | AUC=%.4f  F1=%.4f  DPD=%.4f",
        mid, met["roc_auc"], met["f1"], fmet.get("DPD", 0),
    )
    return {
        "Model":       mid,
        "Description": desc,
        "Accuracy":    round(met["accuracy"],  4),
        "Precision":   round(met["precision"], 4),
        "Recall":      round(met["recall"],    4),
        "F1":          round(met["f1"],        4),
        "ROC_AUC":     round(met["roc_auc"],   4),
        "Bal_Acc":     round(met["bal_acc"],   4),
        "Log_Loss":    round(met["log_loss"],  4),
        "MCC":         round(met["mcc"],       4),
        "DPD":         fmet.get("DPD"),
        "DIR":         fmet.get("DIR"),
        "EOD":         fmet.get("EOD"),
    }


def _group_weights(
    df: pd.DataFrame,
    idx,
    col: str,
) -> np.ndarray:
    """Compute inverse-frequency sample weights for one sensitive column."""
    g = df.loc[idx, col].fillna("Unknown")
    c = g.value_counts()
    n = len(g)
    return g.map(lambda x: n / (len(c) * c[x])).values.astype(float)
