"""
data_loader.py
==============
Load and validate the raw Newspaper Churn CSV.

Usage
-----
    from data_loader import load_data
    df = load_data()                       # uses RAW_DATA_PATH from config
    df = load_data("data/my_file.csv")     # custom path
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COL

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_data(path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load the raw CSV and return a validated DataFrame.

    Parameters
    ----------
    path : str | Path | None
        Override the default RAW_DATA_PATH from config.
    verbose : bool
        Print a dataset summary to stdout.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame exactly as read from CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the resolved path.
    ValueError
        If the target column is missing or contains unexpected values.
    """
    csv_path = Path(path) if path else RAW_DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: '{csv_path}'\n"
            "  → Place 'NewspaperChurn.csv' in the data/ folder.\n"
            "  → Or pass the full path: load_data('/your/path/file.csv')"
        )

    df = pd.read_csv(csv_path)
    _validate(df)

    if verbose:
        _print_summary(df)

    logger.info("Loaded %d rows × %d cols from %s", *df.shape, csv_path.name)
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    """Raise on missing or malformed target column."""
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )
    actual_vals = set(df[TARGET_COL].dropna().unique())
    if not actual_vals.issubset({"YES", "NO"}):
        unexpected = actual_vals - {"YES", "NO"}
        raise ValueError(f"Unexpected target values: {unexpected}")
    logger.info("Schema validation passed.")


def _print_summary(df: pd.DataFrame) -> None:
    """Print a formatted dataset summary."""
    counts = df[TARGET_COL].value_counts()
    pcts   = df[TARGET_COL].value_counts(normalize=True).mul(100)
    print("\n" + "=" * 58)
    print("  DATASET SUMMARY")
    print("=" * 58)
    print(f"  Rows        : {df.shape[0]:,}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Duplicates  : {df.duplicated().sum():,}")
    print(f"  Missing vals: {df.isnull().sum().sum():,}")
    print(f"  Subscriber YES : {counts.get('YES', 0):,}  "
          f"({pcts.get('YES', 0):.2f}%)")
    print(f"  Subscriber NO  : {counts.get('NO', 0):,}  "
          f"({pcts.get('NO', 0):.2f}%)")
    ratio = counts.get("NO", 0) / max(counts.get("YES", 1), 1)
    print(f"  Imbalance ratio: {ratio:.2f}:1")
    print("=" * 58 + "\n")
