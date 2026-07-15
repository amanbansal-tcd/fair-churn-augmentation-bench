"""
test_fairness.py
================
Unit tests for src/fairness.py  (12 tests)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fairness import run_fairness_audit, fairness_summary_df, _group_metrics


@pytest.fixture
def biased_data():
    """
    Synthetic dataset with a deliberately biased model:
    Majority group gets ~50% positive predictions,
    Minority group gets ~10% â€” DPD should be well above 0.20.
    """
    rng   = np.random.default_rng(0)
    n     = 500
    group = np.array(["Majority"] * 300 + ["Minority"] * 200)
    y_true = (rng.uniform(size=n) < 0.25).astype(int)
    y_pred = np.zeros(n, dtype=int)
    y_pred[:300] = (rng.uniform(size=300) < 0.50).astype(int)
    y_pred[300:] = (rng.uniform(size=200) < 0.10).astype(int)
    y_prob = np.concatenate([
        rng.uniform(0.3, 0.9, 300),
        rng.uniform(0.0, 0.3, 200),
    ])
    df  = pd.DataFrame({"SensAttr": group}, index=range(n))
    idx = pd.Index(range(n))
    return y_true, y_pred, y_prob, df, idx


class TestGroupMetrics:
    def test_returns_two_objects(self, biased_data):
        gdf, met = _group_metrics(*biased_data[:3], biased_data[3], biased_data[4], "SensAttr")
        assert isinstance(gdf, pd.DataFrame)
        assert isinstance(met, dict)

    def test_two_groups(self, biased_data):
        gdf, _ = _group_metrics(*biased_data[:3], biased_data[3], biased_data[4], "SensAttr")
        assert len(gdf) == 2

    def test_required_columns_present(self, biased_data):
        gdf, _ = _group_metrics(*biased_data[:3], biased_data[3], biased_data[4], "SensAttr")
        for c in ["Group", "N", "Actual_Rate", "Pred_Pos_Rate", "TPR", "FPR", "FNR"]:
            assert c in gdf.columns

    def test_dpd_positive(self, biased_data):
        _, met = _group_metrics(*biased_data[:3], biased_data[3], biased_data[4], "SensAttr")
        assert met["DPD"] >= 0

    def test_dir_in_zero_one(self, biased_data):
        _, met = _group_metrics(*biased_data[:3], biased_data[3], biased_data[4], "SensAttr")
        assert 0.0 <= met["DIR"] <= 1.0

    def test_known_bias_detected(self, biased_data):
        _, met = _group_metrics(*biased_data[:3], biased_data[3], biased_data[4], "SensAttr")
        assert met["DPD"] > 0.20, f"Expected DPD > 0.20, got {met['DPD']}"

    def test_all_metric_keys_present(self, biased_data):
        _, met = _group_metrics(*biased_data[:3], biased_data[3], biased_data[4], "SensAttr")
        for k in ["DPD", "DPR", "DIR", "EOD", "EqOD", "FPR_D", "FNR_D", "SPD"]:
            assert k in met

    def test_small_groups_excluded(self, biased_data):
        y_t, y_p, y_pr, df, _ = biased_data
        tiny_df  = pd.concat([
            df,
            pd.DataFrame({"SensAttr": ["Tiny"] * 5}, index=range(500, 505))
        ])
        tiny_idx = pd.Index(range(505))
        y_t2  = np.concatenate([y_t, np.zeros(5, int)])
        y_p2  = np.concatenate([y_p, np.zeros(5, int)])
        y_pr2 = np.concatenate([y_pr, np.zeros(5)])
        gdf, _ = _group_metrics(y_t2, y_p2, y_pr2, tiny_df, tiny_idx, "SensAttr")
        assert "Tiny" not in gdf["Group"].values


class TestRunFairnessAudit:
    def test_audit_returns_dict(self, biased_data):
        y_t, y_p, y_pr, df, idx = biased_data
        audit = run_fairness_audit(y_t, y_p, y_pr, df, idx, attrs=["SensAttr"])
        assert isinstance(audit, dict)
        assert "SensAttr" in audit

    def test_audit_structure(self, biased_data):
        y_t, y_p, y_pr, df, idx = biased_data
        audit = run_fairness_audit(y_t, y_p, y_pr, df, idx, attrs=["SensAttr"])
        assert "groups"  in audit["SensAttr"]
        assert "metrics" in audit["SensAttr"]

    def test_missing_attr_skipped(self, biased_data):
        y_t, y_p, y_pr, df, idx = biased_data
        audit = run_fairness_audit(y_t, y_p, y_pr, df, idx,
                                   attrs=["SensAttr", "NonExistent"])
        assert "NonExistent" not in audit


class TestFairnessSummaryDf:
    def test_returns_dataframe(self, biased_data):
        y_t, y_p, y_pr, df, idx = biased_data
        audit   = run_fairness_audit(y_t, y_p, y_pr, df, idx, attrs=["SensAttr"])
        summary = fairness_summary_df(audit)
        assert isinstance(summary, pd.DataFrame)
        assert "Attribute" in summary.columns
        assert "DPD" in summary.columns
