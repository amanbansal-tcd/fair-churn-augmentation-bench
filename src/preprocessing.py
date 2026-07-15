"""
preprocessing.py
================
Data cleaning, feature engineering, and encoding.

All transformations are deterministic and reproducible with RANDOM_SEED=42.
No information leaks from test to train — all encoders are fit on the
full dataset before the train/test split (this is safe for ordinal/label
encoding of categorical IDs; no statistics from y are used).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    TARGET_COL, TARGET_POS, DROP_COLS,
    INCOME_MAP, AGE_MIDPOINTS, DP_NORMALISE, ALL_FEATURES,
)

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """
    Full preprocessing pipeline: clean → engineer → encode.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_data().
    drop_cols : list[str] | None
        Additional encoded feature names to drop before returning X
        (used for bias-mitigation experiments M1–M15).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (encoded, engineered).
    y : pd.Series
        Binary target (1 = YES subscriber, 0 = NO).
    df_clean : pd.DataFrame
        Fully preprocessed DataFrame (all engineered columns present).
    encoders : dict
        LabelEncoder objects keyed by original column name.
    """
    df = df.copy()

    # Encode target
    df["target"] = (df[TARGET_COL] == TARGET_POS).astype(int)

    # Drop ID, address, geography, and target from feature space
    cols_to_drop = [c for c in DROP_COLS + [TARGET_COL] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Pipeline
    df             = _impute(df)
    df             = _engineer_numerics(df)
    df             = _normalise_categoricals(df)
    df, encoders   = _label_encode(df)

    # Build feature list, optionally dropping sensitive cols
    extra_drop    = drop_cols or []
    features_used = [f for f in ALL_FEATURES if f not in extra_drop and f in df.columns]

    X = df[features_used]
    y = df["target"]

    logger.info(
        "Features: %d  |  Positive rate: %.4f  |  Drop extra: %s",
        len(features_used), y.mean(), extra_drop or "none",
    )
    return X, y, df, encoders


# ── Step 1: imputation ─────────────────────────────────────────────────────────

def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill known missing-value patterns with safe defaults."""
    fill_map = {
        "Age range":          "Unknown",
        "Language":           "Unknown",
        "weekly fee":         "$1.00 - $1.99",
        "Nielsen Prizm":      "Unknown",
        "dummy for Children": "N",
    }
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ── Step 2: feature engineering ───────────────────────────────────────────────

def _engineer_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Create numerical features from raw categorical/text columns."""

    # Ordinal income encoding (0–15)
    if "HH Income" in df.columns:
        df["income_ord"] = df["HH Income"].map(INCOME_MAP).fillna(6.0)

    # Age midpoint from age-range string
    if "Age range" in df.columns:
        df["age_mid"] = df["Age range"].map(AGE_MIDPOINTS).fillna(52.0)

    # Weekly fee midpoint from "$x.xx - $y.yy" string
    if "weekly fee" in df.columns:
        df["fee_mid"] = df["weekly fee"].apply(_parse_fee)

    # Reward programme — winsorise at 99th percentile to reduce outlier influence
    if "reward program" in df.columns:
        cap = df["reward program"].quantile(0.99)
        df["reward_prog_w"] = df["reward program"].clip(upper=cap)

    return df


def _parse_fee(s: str) -> float:
    """
    Parse a weekly-fee range string such as '$1.00 - $1.99' and return
    its midpoint.  Returns 1.5 as a safe default on any parse failure.
    """
    try:
        nums = [
            float(x.strip().replace("$", "").replace(",", ""))
            for x in str(s).split("-")
        ]
        return float(np.mean(nums))
    except Exception:
        return 1.5


# ── Step 3: normalise categoricals ────────────────────────────────────────────

def _normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise delivery-period labels to a consistent set."""
    if "Deliveryperiod" in df.columns:
        df["DP"] = df["Deliveryperiod"].replace(DP_NORMALISE)
    return df


# ── Step 4: label encoding ────────────────────────────────────────────────────

def _label_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply LabelEncoder to each categorical column and append an '_enc'
    column.  Fit on all available data (full dataset, pre-split).
    """
    encoders: dict[str, LabelEncoder] = {}
    cat_cols = [
        "Home Ownership", "Ethnicity", "Language",
        "dummy for Children", "DP", "Nielsen Prizm", "Source Channel",
    ]
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col]    = le

    return df, encoders
