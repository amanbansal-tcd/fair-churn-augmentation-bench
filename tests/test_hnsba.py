"""
test_hnsba.py
=============
Unit tests for src/augmentation/hnsba.py covering the profiler and the
validate_and_repair step, without any network calls to the Anthropic API.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.augmentation.hnsba import (
    compute_marginal_profile,
    validate_and_repair,
    run_hnsba_augmentation,
)
from src.config import ALL_FEATURES


@pytest.fixture
def minority_df():
    np.random.seed(0)
    n = 40
    return pd.DataFrame({
        "income_ord": np.random.randint(0, 16, n).astype(float),
        "age_mid": np.random.uniform(20, 75, n),
        "fee_mid": np.random.uniform(0.5, 5.0, n),
        "Year Of Residence": np.random.randint(0, 30, n).astype(float),
        "reward_prog_w": np.random.uniform(0, 200, n),
        "Home Ownership_enc": np.random.randint(0, 3, n).astype(float),
        "Ethnicity_enc": np.random.randint(0, 10, n).astype(float),
        "Language_enc": np.random.randint(0, 5, n).astype(float),
        "dummy for Children_enc": np.random.randint(0, 2, n).astype(float),
        "DP_enc": np.random.randint(0, 6, n).astype(float),
        "Nielsen Prizm_enc": np.random.randint(0, 9, n).astype(float),
        "Source Channel_enc": np.random.randint(0, 8, n).astype(float),
    })


class TestComputeMarginalProfile:
    def test_all_columns_present(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        assert set(profile.keys()) == set(minority_df.columns)

    def test_int_columns_have_categories(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        assert "categories" in profile["Ethnicity_enc"]
        assert "categories" not in profile["age_mid"]

    def test_min_max_within_bounds(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        for col in minority_df.columns:
            assert profile[col]["min"] <= minority_df[col].min()
            assert profile[col]["max"] >= minority_df[col].max()


class TestValidateAndRepair:
    def test_out_of_range_values_are_clipped(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        columns = list(minority_df.columns)
        bad_row = {c: 1e9 for c in columns}
        repaired = validate_and_repair([bad_row], profile, columns)
        assert len(repaired) == 1
        for col in columns:
            assert repaired.iloc[0][col] <= profile[col]["max"]

    def test_out_of_vocabulary_category_snapped(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        columns = list(minority_df.columns)
        row = {c: float(minority_df[c].iloc[0]) for c in columns}
        row["Ethnicity_enc"] = 999.0  # not an observed category code
        repaired = validate_and_repair([row], profile, columns)
        assert int(repaired.iloc[0]["Ethnicity_enc"]) in profile["Ethnicity_enc"]["categories"]

    def test_missing_column_row_is_dropped(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        columns = list(minority_df.columns)
        incomplete_row = {c: 1.0 for c in columns[:-1]}  # missing last column
        repaired = validate_and_repair([incomplete_row], profile, columns)
        assert len(repaired) == 0

    def test_integer_columns_are_whole_numbers(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        columns = list(minority_df.columns)
        row = {c: float(minority_df[c].iloc[0]) + 0.4 for c in columns}
        repaired = validate_and_repair([row], profile, columns)
        assert (repaired["income_ord"] == repaired["income_ord"].round()).all()

    def test_output_has_expected_columns(self, minority_df):
        profile = compute_marginal_profile(minority_df)
        columns = list(minority_df.columns)
        row = {c: float(minority_df[c].iloc[0]) for c in columns}
        repaired = validate_and_repair([row], profile, columns)
        assert list(repaired.columns) == columns


class TestRunHnsbaAugmentationSkipsWithoutKey:
    def test_returns_none_without_api_key(self, minority_df, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        X = pd.concat([minority_df, minority_df], ignore_index=True)
        y = pd.Series([1] * len(minority_df) + [0] * len(minority_df), name="target")
        result = run_hnsba_augmentation(X, y, n_synth=5)
        assert result is None
