"""
test_evaluation.py
==================
Unit tests for src/evaluation.py  (9 tests)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_model, metrics_dataframe


@pytest.fixture
def perfect():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    return y, y.copy(), y.astype(float)


@pytest.fixture
def random_preds():
    rng  = np.random.default_rng(42)
    y    = rng.integers(0, 2, 200)
    prob = rng.uniform(0, 1, 200)
    return y, (prob > 0.5).astype(int), prob


class TestEvaluateModel:
    def test_returns_required_keys(self, random_preds):
        res = evaluate_model(*random_preds)
        for k in ["accuracy","precision","recall","f1","roc_auc",
                  "pr_auc","bal_acc","log_loss","mcc","y_pred","y_prob"]:
            assert k in res

    def test_perfect_auc_one(self, perfect):
        assert evaluate_model(*perfect)["roc_auc"] == pytest.approx(1.0)

    def test_perfect_f1_one(self, perfect):
        assert evaluate_model(*perfect)["f1"] == pytest.approx(1.0)

    def test_metrics_in_valid_range(self, random_preds):
        res = evaluate_model(*random_preds)
        for k in ["accuracy","precision","recall","f1","roc_auc","pr_auc","bal_acc"]:
            assert 0.0 <= res[k] <= 1.0

    def test_log_loss_positive(self, random_preds):
        assert evaluate_model(*random_preds)["log_loss"] > 0

    def test_mcc_in_range(self, random_preds):
        assert -1.0 <= evaluate_model(*random_preds)["mcc"] <= 1.0

    def test_y_pred_correct_length(self, random_preds):
        res = evaluate_model(*random_preds)
        assert len(res["y_pred"]) == len(random_preds[0])


class TestMetricsDataframe:
    def test_returns_dataframe(self, random_preds):
        m  = evaluate_model(*random_preds)
        df = metrics_dataframe({"A": m, "B": m})
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns

    def test_sorted_by_roc_auc(self, random_preds):
        m    = evaluate_model(*random_preds)
        df   = metrics_dataframe({"A": m, "B": m})
        aucs = df["roc_auc"].tolist()
        assert aucs == sorted(aucs, reverse=True)
