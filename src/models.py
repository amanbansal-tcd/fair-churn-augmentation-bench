"""
models.py
=========
Model definitions, training, cross-validation, and persistence.

Five baseline classifiers are supported:
  • Logistic Regression  (sklearn Pipeline + StandardScaler)
  • Random Forest        (sklearn)
  • XGBoost
  • LightGBM
  • CatBoost
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

from src.config import (
    RANDOM_SEED, CV_FOLDS,
    LR_PARAMS, RF_PARAMS, XGB_PARAMS, LGB_PARAMS, CAT_PARAMS,
    MODEL_DIR,
)
from src.evaluation import evaluate_model

logger = logging.getLogger(__name__)


def _fallback_gradient_boosting(**kwargs):
    """Provide a lightweight sklearn fallback when optional boosters are unavailable."""
    return HistGradientBoostingClassifier(
        max_depth=kwargs.get("max_depth", 6),
        learning_rate=kwargs.get("learning_rate", 0.05),
        max_iter=kwargs.get("n_estimators", 400),
        random_state=RANDOM_SEED,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_models(scale_pos_weight: float) -> dict:
    """
    Instantiate all five baseline classifiers with class-imbalance
    correction applied consistently.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio (n_negative / n_positive) used by XGBoost and CatBoost.
        Logistic Regression, Random Forest, and LightGBM use
        class_weight='balanced' instead.

    Returns
    -------
    dict
        Mapping {model_name: unfitted_estimator}.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(class_weight="balanced", **LR_PARAMS)),
        ]),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", **RF_PARAMS
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight, **XGB_PARAMS
            ) if xgb is not None else _fallback_gradient_boosting(**XGB_PARAMS)
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight="balanced", **LGB_PARAMS
        ),
        "CatBoost": (
            CatBoostClassifier(
                class_weights=[1, scale_pos_weight], **CAT_PARAMS
            ) if CatBoostClassifier is not None else _fallback_gradient_boosting(**CAT_PARAMS)
        ),
    }


def train_all(
    models: dict,
    X_train, X_test,
    y_train, y_test,
) -> dict:
    """
    Fit every model on the training set and evaluate on the test set.

    Returns
    -------
    dict
        {model_name: metrics_dict} — see evaluate_model() for keys.
    """
    results = {}
    for name, model in models.items():
        logger.info("  Training %-22s ...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate_model(y_test, y_pred, y_prob)
        logger.info(
            "    AUC=%.4f  F1=%.4f  BalAcc=%.4f",
            results[name]["roc_auc"],
            results[name]["f1"],
            results[name]["bal_acc"],
        )
    return results


def cross_validate_model(
    model,
    X, y,
    scoring: list[str] | None = None,
) -> dict:
    """
    Run stratified k-fold cross-validation on the training set.

    Returns
    -------
    dict
        {metric: {"mean": float, "std": float}}
    """
    scoring = scoring or ["roc_auc", "f1", "balanced_accuracy"]
    cv      = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    raw = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    summary = {}
    for s in scoring:
        vals          = raw[f"test_{s}"]
        summary[s]    = {"mean": round(vals.mean(), 4), "std": round(vals.std(), 4)}
        logger.info("  CV %-22s : %.4f ± %.4f", s, vals.mean(), vals.std())
    return summary


def save_model(model, name: str, out_dir: Path | None = None) -> Path:
    """Persist a fitted model with joblib."""
    out  = Path(out_dir) if out_dir else MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str, model_dir: Path | None = None):
    """Load a previously saved model."""
    out  = Path(model_dir) if model_dir else MODEL_DIR
    path = out / f"{name.lower().replace(' ', '_')}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
