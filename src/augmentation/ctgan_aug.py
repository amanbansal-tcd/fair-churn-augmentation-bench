"""
ctgan_aug.py
============
CTGAN (Conditional Tabular GAN) minority-class augmentation.

Split out from classical.py so that the heavy `ctgan` dependency (and its
torch backend) is only imported when this technique is actually used —
callers running `--no-ctgan` never pay the import cost.

Design notes
------------
- CTGAN is trained on the minority class only, then sampled for n_synth
  new synthetic rows.
- Post-processing clips every feature to its observed [min, max] range
  (from FEATURE_RANGES in config.py) and rounds integer-encoded columns —
  CTGAN's continuous generator can otherwise emit out-of-range or
  fractional values for what are really discrete/ordinal columns.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED, FEATURE_RANGES,
    AUG_CTGAN_EPOCHS, AUG_CTGAN_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


def ctgan_augment(
    X: pd.DataFrame,
    y: pd.Series,
    n_synth: int = 3_570,
    epochs: int = 150,
    batch_size: int = 500,
    random_state: int = 42,
) -> tuple:
    """
    CTGAN-based minority class augmentation.

    Parameters
    ----------
    X : pd.DataFrame
        Full training feature matrix.
    y : pd.Series
        Binary training target (0/1).
    n_synth : int
        Number of synthetic minority rows to generate.
    epochs : int
        CTGAN training epochs.
    batch_size : int
        CTGAN batch size.
    random_state : int
        Controls the shuffle after concatenation.

    Returns
    -------
    (X_combined, y_combined) : tuple[pd.DataFrame, pd.Series]
        Original training data + synthetic minority rows, shuffled.

    Raises
    ------
    ImportError
        If the `ctgan` package is not installed.
    """
    try:
        from ctgan import CTGAN as _CTGAN
    except Exception:
        raise ImportError("CTGAN is not installed. Run: pip install ctgan")

    X_min = X[y == 1].copy()
    logger.info("  CTGAN training on %d minority samples...", len(X_min))

    safe_batch_size = max(1, min(batch_size, len(X_min)))
    model = _CTGAN(epochs=epochs, batch_size=safe_batch_size, pac=1, verbose=False)
    model.fit(X_min.astype(float))

    X_synth_raw = model.sample(n_synth)
    X_synth = pd.DataFrame(X_synth_raw, columns=X.columns)
    X_synth = _clip_and_round(X_synth, X)

    X_combined = pd.concat([X.reset_index(drop=True), X_synth.reset_index(drop=True)], ignore_index=True)
    y_synth = pd.Series(np.ones(n_synth, dtype=int), name=y.name)
    y_combined = pd.concat([y.reset_index(drop=True), y_synth], ignore_index=True)

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(X_combined))
    return (
        X_combined.iloc[perm].reset_index(drop=True),
        y_combined.iloc[perm].reset_index(drop=True),
    )


def regen_ctgan_synthetic(orig_X: pd.DataFrame, orig_y: pd.Series, n_new: int) -> tuple:
    """
    Deterministically regenerate the CTGAN synthetic-only rows for CSV export.

    Used by classical.py's export_augmented_csvs(), which cannot recover the
    synthetic subset by positional slicing because the combined CTGAN
    dataset is shuffled.
    """
    try:
        from ctgan import CTGAN as _CTGAN
    except Exception:
        raise ImportError("CTGAN not installed: pip install ctgan")

    X_min = orig_X[orig_y == 1].copy()
    safe_batch_size = max(1, min(AUG_CTGAN_BATCH_SIZE, len(X_min)))
    model = _CTGAN(epochs=AUG_CTGAN_EPOCHS, batch_size=safe_batch_size, pac=1, verbose=False)
    model.fit(X_min.astype(float))
    raw = model.sample(n_new)
    X_synth = pd.DataFrame(raw, columns=orig_X.columns)
    X_synth = _clip_and_round(X_synth, orig_X)

    y_synth = pd.Series(np.ones(n_new, dtype=int), name=orig_y.name)
    return X_synth, y_synth


def _clip_and_round(X_synth: pd.DataFrame, X_ref: pd.DataFrame) -> pd.DataFrame:
    """Clip synthetic columns to the observed [min, max] range and round integer columns."""
    for col in X_ref.columns:
        cfg = FEATURE_RANGES.get(col, {})
        col_min = cfg.get("min", float(X_ref[col].min()))
        col_max = cfg.get("max", float(X_ref[col].max()))
        X_synth[col] = X_synth[col].clip(lower=col_min, upper=col_max)
        if cfg.get("type") == "int":
            X_synth[col] = X_synth[col].round().astype(int)
    return X_synth
