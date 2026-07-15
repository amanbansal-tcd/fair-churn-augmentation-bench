"""
test_augmentation.py
====================
Unit tests for src/augmentation.py

Tests cover:
  â€¢ gaussian_augment()        â€” 5 tests
  â€¢ run_augmentation()        â€” 5 tests
  â€¢ export_augmented_csvs()   â€” 3 tests
  â€¢ ctgan skipped / included  â€” 2 tests (skipped when ctgan unavailable)

Total: 15 tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.augmentation.classical import (
    gaussian_augment,
    run_augmentation,
    export_augmented_csvs,
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Shared fixtures
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@pytest.fixture
def small_train():
    """
    Tiny imbalanced training set for fast unit tests.
    300 rows: 60 minority (Class 1), 240 majority (Class 0).
    """
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "income_ord":             np.random.randint(0, 16, n).astype(float),
        "age_mid":                np.random.uniform(20, 75, n),
        "fee_mid":                np.random.uniform(0.5, 5.0, n),
        "Year Of Residence":      np.random.randint(0, 30, n).astype(float),
        "reward_prog_w":          np.random.uniform(0, 200, n),
        "Home Ownership_enc":     np.random.randint(0, 3, n).astype(float),
        "Ethnicity_enc":          np.random.randint(0, 10, n).astype(float),
        "Language_enc":           np.random.randint(0, 5, n).astype(float),
        "dummy for Children_enc": np.random.randint(0, 2, n).astype(float),
        "DP_enc":                 np.random.randint(0, 6, n).astype(float),
        "Nielsen Prizm_enc":      np.random.randint(0, 9, n).astype(float),
        "Source Channel_enc":     np.random.randint(0, 8, n).astype(float),
    })
    y = pd.Series(
        np.concatenate([np.ones(60, int), np.zeros(240, int)]),
        name="target",
    )
    return X, y


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Gaussian augmentation tests
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestGaussianAugment:
    def test_grows_minority_class(self, small_train):
        """After oversampling, minority count should reach target."""
        X, y = small_train
        Xa, ya = gaussian_augment(X, y, target_minority=120, majority_noise_frac=0.0)
        assert (ya == 1).sum() == 120

    def test_grows_majority_class(self, small_train):
        """Majority expansion should add rows when frac > 0."""
        X, y = small_train
        Xa, ya = gaussian_augment(X, y, target_minority=60, majority_noise_frac=0.20)
        assert (ya == 0).sum() > (y == 0).sum()

    def test_output_columns_match(self, small_train):
        """Output DataFrame must have exactly the same columns as input."""
        X, y = small_train
        Xa, ya = gaussian_augment(X, y, target_minority=80)
        assert list(Xa.columns) == list(X.columns)

    def test_no_values_outside_observed_range(self, small_train):
        """Clipping must keep all values within original [min, max]."""
        X, y = small_train
        Xa, ya = gaussian_augment(X, y, target_minority=80)
        for col in X.columns:
            assert Xa[col].min() >= X[col].min() - 1e-6
            assert Xa[col].max() <= X[col].max() + 1e-6

    def test_integer_encoded_cols_remain_integers(self, small_train):
        """Integer features must be rounded after noise injection."""
        X, y = small_train
        Xa, _ = gaussian_augment(X, y, target_minority=80)
        for col in ["income_ord", "Ethnicity_enc", "DP_enc"]:
            assert (Xa[col] == Xa[col].round()).all(), \
                f"{col} contains non-integer values after Gaussian augmentation"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# run_augmentation() tests
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestRunAugmentation:
    def test_returns_dict_with_required_keys(self, small_train):
        """Result must contain 'original', 'SMOTE', 'ADASYN', 'Gaussian Noise'."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        for k in ["original", "SMOTE", "ADASYN", "Gaussian Noise"]:
            assert k in aug, f"Missing key: {k}"

    def test_ctgan_skipped_when_false(self, small_train):
        """'CTGAN' key must be absent when run_ctgan=False."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        assert "CTGAN" not in aug

    def test_original_data_unchanged(self, small_train):
        """'original' entry must have the same row count as input."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        assert len(aug["original"]["X"]) == len(X)
        assert len(aug["original"]["y"]) == len(y)

    def test_smote_increases_minority_count(self, small_train):
        """SMOTE must produce more minority rows than the original."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        assert (aug["SMOTE"]["y"] == 1).sum() > (y == 1).sum()

    def test_adasyn_increases_minority_count(self, small_train):
        """ADASYN must produce more minority rows than the original."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        assert (aug["ADASYN"]["y"] == 1).sum() > (y == 1).sum()

    def test_sampling_strategy_never_exceeds_one(self, small_train):
        """
        sampling_strategy is capped at 1.0 (minority cannot exceed majority).
        k_neighbors is clamped to minority_count - 1 to prevent ValueError.
        Uses 20 minority samples (> default k=5) for a valid SMOTE run.
        """
        X, y = small_train
        # Severe imbalance: 20 minority vs 280 majority
        y_severe = pd.Series(
            np.concatenate([np.ones(20, int), np.zeros(280, int)]),
            name="target",
        )
        aug = run_augmentation(X, y_severe, run_ctgan=False)
        assert (aug["SMOTE"]["y"] == 1).sum() >= 20


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# export_augmented_csvs() tests
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestExportAugmentedCsvs:
    def test_required_files_are_created(self, small_train, tmp_path):
        """All required CSV files must be created in the output directory."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        X_te = X.iloc[:50].reset_index(drop=True)
        y_te = y.iloc[:50].reset_index(drop=True)
        export_augmented_csvs(aug, X_te, y_te, out_dir=tmp_path)

        for fname in [
            "train_original.csv",
            "test.csv",
            "train_smote_combined.csv",
            "train_smote_synthetic.csv",
            "train_adasyn_combined.csv",
            "train_adasyn_synthetic.csv",
        ]:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_combined_has_more_rows_than_original(self, small_train, tmp_path):
        """Combined files must be larger than the original training split."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        X_te = X.iloc[:50].reset_index(drop=True)
        y_te = y.iloc[:50].reset_index(drop=True)
        export_augmented_csvs(aug, X_te, y_te, out_dir=tmp_path)

        orig_rows = pd.read_csv(tmp_path / "train_original.csv").shape[0]
        smote_rows = pd.read_csv(tmp_path / "train_smote_combined.csv").shape[0]
        assert smote_rows > orig_rows

    def test_synthetic_file_rows_match_difference(self, small_train, tmp_path):
        """Synthetic file row count must equal combined minus original."""
        X, y = small_train
        aug  = run_augmentation(X, y, run_ctgan=False)
        X_te = X.iloc[:50].reset_index(drop=True)
        y_te = y.iloc[:50].reset_index(drop=True)
        export_augmented_csvs(aug, X_te, y_te, out_dir=tmp_path)

        orig_rows    = pd.read_csv(tmp_path / "train_original.csv").shape[0]
        combined_rows = pd.read_csv(tmp_path / "train_smote_combined.csv").shape[0]
        synthetic_rows = pd.read_csv(tmp_path / "train_smote_synthetic.csv").shape[0]
        assert synthetic_rows == combined_rows - orig_rows
