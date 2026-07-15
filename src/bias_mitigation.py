"""
bias_mitigation.py
==================
M0–M15 feature-removal bias mitigation experiments.

Each variant trains LightGBM with a specific subset of features
(sensitive columns progressively removed) and reports both predictive
performance and a fairness metric (DPD on Ethnicity by default).
"""

import logging
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from src.config import ALL_FEATURES, SENSITIVE_FEATS, LGB_PARAMS, REPORT_DIR
from src.evaluation import evaluate_model
from src.fairness import _group_metrics

logger = logging.getLogger(__name__)

# Ordered list of (model_id, description, columns_to_drop)
VARIANTS: list[tuple[str, str, list[str]]] = [
    ("M0",  "Baseline (all features)",           []),
    ("M1",  "Remove Ethnicity",                  SENSITIVE_FEATS["ETHNICITY"]),
    ("M2",  "Remove Language",                   SENSITIVE_FEATS["LANGUAGE"]),
    ("M3",  "Remove Home Ownership",             SENSITIVE_FEATS["HOMEOWN"]),
    ("M4",  "Remove Geography",                  SENSITIVE_FEATS["GEOGRAPHY"]),
    ("M5",  "Remove Eth + Lang",                 SENSITIVE_FEATS["ETHNICITY"] + SENSITIVE_FEATS["LANGUAGE"]),
    ("M6",  "Remove Eth + Geo",                  SENSITIVE_FEATS["ETHNICITY"] + SENSITIVE_FEATS["GEOGRAPHY"]),
    ("M7",  "Remove Lang + Geo",                 SENSITIVE_FEATS["LANGUAGE"]  + SENSITIVE_FEATS["GEOGRAPHY"]),
    ("M8",  "Remove HO + Geo",                   SENSITIVE_FEATS["HOMEOWN"]   + SENSITIVE_FEATS["GEOGRAPHY"]),
    ("M9",  "Remove Eth + Lang + Geo",           SENSITIVE_FEATS["ETHNICITY"] + SENSITIVE_FEATS["LANGUAGE"]  + SENSITIVE_FEATS["GEOGRAPHY"]),
    ("M10", "Remove Eth + HO",                   SENSITIVE_FEATS["ETHNICITY"] + SENSITIVE_FEATS["HOMEOWN"]),
    ("M11", "Remove Lang + HO",                  SENSITIVE_FEATS["LANGUAGE"]  + SENSITIVE_FEATS["HOMEOWN"]),
    ("M12", "Remove Eth + Lang + HO",            SENSITIVE_FEATS["ETHNICITY"] + SENSITIVE_FEATS["LANGUAGE"]  + SENSITIVE_FEATS["HOMEOWN"]),
    ("M13", "Remove Eth + HO + Geo",             SENSITIVE_FEATS["ETHNICITY"] + SENSITIVE_FEATS["HOMEOWN"]   + SENSITIVE_FEATS["GEOGRAPHY"]),
    ("M14", "Remove Lang + HO + Geo",            SENSITIVE_FEATS["LANGUAGE"]  + SENSITIVE_FEATS["HOMEOWN"]   + SENSITIVE_FEATS["GEOGRAPHY"]),
    ("M15", "Remove ALL sensitive",              SENSITIVE_FEATS["ETHNICITY"] + SENSITIVE_FEATS["LANGUAGE"]  + SENSITIVE_FEATS["HOMEOWN"]   + SENSITIVE_FEATS["GEOGRAPHY"]),
]


def run_mitigation_experiments(
    df: pd.DataFrame,
    X_train, X_test,
    y_train, y_test,
    fairness_attr: str = "Ethnicity",
) -> pd.DataFrame:
    """
    Train M0–M15 LightGBM variants and return a results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame (needed for fairness group look-up).
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    fairness_attr : str
        Sensitive attribute used for fairness metrics.

    Returns
    -------
    pd.DataFrame
        One row per variant with performance + fairness columns.
    """
    rows = []
    for mid, desc, drop in VARIANTS:
        feats = [f for f in ALL_FEATURES if f not in drop and f in X_train.columns]
        m     = lgb.LGBMClassifier(class_weight="balanced", **LGB_PARAMS)
        m.fit(X_train[feats], y_train)

        yp   = m.predict(X_test[feats])
        ypr  = m.predict_proba(X_test[feats])[:, 1]
        met  = evaluate_model(y_test, yp, ypr)
        _, fmet = _group_metrics(
            y_test.values, yp, ypr, df, X_test.index, fairness_attr
        )

        rows.append({
            "Model":       mid,
            "Description": desc,
            "Features":    len(feats),
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
        })
        logger.info(
            "%s | AUC=%.4f  F1=%.4f  DPD=%.4f",
            mid, met["roc_auc"], met["f1"], fmet.get("DPD", 0),
        )

    df_out = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(REPORT_DIR / "mitigation_results.csv", index=False)
    logger.info("Mitigation results saved (%d variants).", len(df_out))
    return df_out
