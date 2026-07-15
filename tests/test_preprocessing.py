"""
test_preprocessing.py
=====================
Unit tests for src/preprocessing.py  (19 tests)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_features, _parse_fee,
    _impute, _engineer_numerics, _normalise_categoricals,
)


@pytest.fixture
def sample_df():
    """Minimal synthetic dataset that mirrors the real CSV schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "SubscriptionID":     range(n),
        "HH Income":          np.random.choice(
            ["Under $20,000", "$  30,000 - $39,999", "$100,000 - $124,999"], n
        ),
        "Home Ownership":     np.random.choice(["OWNER", "RENTER"], n),
        "Ethnicity":          np.random.choice(["Hispanic", "English", "Chinese"], n),
        "dummy for Children": np.random.choice(["Y", "N"], n),
        "Year Of Residence":  np.random.randint(1, 30, n),
        "Age range":          np.random.choice(["35-39", "50-54", "65-69", None], n),
        "Language":           np.random.choice(["English", "Spanish", None], n),
        "Address":            ["123 Main St"] * n,
        "State":              ["CA"] * n,
        "City":               ["LA"] * n,
        "County":             ["LA Co"] * n,
        "Zip Code":           [90001] * n,
        "weekly fee":         np.random.choice(["$1.00 - $1.99", "$3.00 - $3.99", None], n),
        "Deliveryperiod":     np.random.choice(["7Day", "SunOnly", "SoooTFS"], n),
        "Nielsen Prizm":      np.random.choice(["FM", "MW", None], n),
        "reward program":     np.random.randint(0, 100, n),
        "Source Channel":     np.random.choice(["Partner", "CustCall"], n),
        "Subscriber":         np.random.choice(["YES", "NO"], n, p=[0.2, 0.8]),
    })


class TestBuildFeatures:
    def test_returns_four_objects(self, sample_df):
        X, y, df, enc = build_features(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(enc, dict)

    def test_target_is_binary(self, sample_df):
        _, y, _, _ = build_features(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_no_missing_in_X(self, sample_df):
        X, _, _, _ = build_features(sample_df)
        assert X.isnull().sum().sum() == 0

    def test_positive_rate_in_range(self, sample_df):
        _, y, _, _ = build_features(sample_df)
        assert 0.05 < y.mean() < 0.95

    def test_drop_cols_removes_feature(self, sample_df):
        X_full, _, _, _ = build_features(sample_df)
        X_drop, _, _, _ = build_features(sample_df, drop_cols=["Ethnicity_enc"])
        assert "Ethnicity_enc" not in X_drop.columns
        assert len(X_full.columns) == len(X_drop.columns) + 1

    def test_id_and_target_excluded(self, sample_df):
        X, _, _, _ = build_features(sample_df)
        assert "SubscriptionID" not in X.columns
        assert "Subscriber" not in X.columns

    def test_all_features_present(self, sample_df):
        from src.config import ALL_FEATURES
        X, _, _, _ = build_features(sample_df)
        for f in ALL_FEATURES:
            assert f in X.columns, f"{f} missing from X"


class TestImpute:
    def test_fills_age_range(self, sample_df):
        sample_df["Age range"] = None
        df_out = _impute(sample_df.copy())
        assert df_out["Age range"].isnull().sum() == 0

    def test_fills_language(self, sample_df):
        sample_df["Language"] = None
        df_out = _impute(sample_df.copy())
        assert df_out["Language"].isnull().sum() == 0


class TestEngineerNumerics:
    def test_income_ord_range(self, sample_df):
        df_out = _engineer_numerics(_impute(sample_df.copy()))
        assert "income_ord" in df_out.columns
        assert df_out["income_ord"].between(0, 15).all()

    def test_age_mid_positive(self, sample_df):
        df_out = _engineer_numerics(_impute(sample_df.copy()))
        assert "age_mid" in df_out.columns
        assert (df_out["age_mid"] > 0).all()

    def test_fee_mid_non_negative(self, sample_df):
        sample_df["weekly fee"] = sample_df["weekly fee"].fillna("$1.00 - $1.99")
        df_out = _engineer_numerics(sample_df.copy())
        assert (df_out["fee_mid"] >= 0).all()

    def test_reward_winsorised(self, sample_df):
        df_out = _engineer_numerics(sample_df.copy())
        cap    = sample_df["reward program"].quantile(0.99)
        assert df_out["reward_prog_w"].max() <= cap + 1e-6


class TestParseFee:
    def test_normal_range(self):
        assert _parse_fee("$1.00 - $1.99") == pytest.approx(1.495)

    def test_zero_range(self):
        assert _parse_fee("$0.00 - $0.01") == pytest.approx(0.005)

    def test_invalid_returns_default(self):
        assert _parse_fee(None) == 1.5
        assert _parse_fee("N/A") == 1.5


class TestNormaliseCategoricals:
    def test_dp_normalised(self, sample_df):
        df_out = _normalise_categoricals(sample_df.copy())
        assert "DP" in df_out.columns
        mask   = sample_df["Deliveryperiod"] == "SoooTFS"
        assert (df_out.loc[mask, "DP"] == "Thu-Sun").all()
