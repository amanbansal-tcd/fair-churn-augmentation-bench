"""
test_llm_aug.py
================
Unit tests for src/augmentation/llm_aug.py's parsing/validation/fallback
logic. No real Ollama server is contacted.
"""

import pytest

from src.augmentation.llm_aug import (
    extract_rows,
    validate_rows,
    compute_stats,
    python_smote,
    python_gaussian_noise,
    is_ollama_reachable,
)

COLUMNS = ["income_ord", "age_mid", "target"]


@pytest.fixture
def minority_rows():
    return [
        {"income_ord": "5", "age_mid": "40.0", "target": "1"},
        {"income_ord": "6", "age_mid": "45.0", "target": "1"},
        {"income_ord": "7", "age_mid": "50.0", "target": "1"},
        {"income_ord": "4", "age_mid": "35.0", "target": "1"},
    ]


@pytest.fixture
def stats(minority_rows):
    return compute_stats(minority_rows, COLUMNS)


class TestExtractRows:
    def test_extracts_fenced_csv_block(self):
        text = "```csv\n5,40.0,1\n6,45.0,1\n```"
        rows = extract_rows(text, n_cols=3)
        assert rows == [["5", "40.0", "1"], ["6", "45.0", "1"]]

    def test_extracts_bare_numeric_lines(self):
        text = "5,40.0,1\n6,45.0,1\n"
        rows = extract_rows(text, n_cols=3)
        assert len(rows) == 2

    def test_rejects_alphabetic_lines(self):
        text = "5,40.0,1\nhello,world,x\n"
        rows = extract_rows(text, n_cols=3)
        assert rows == [["5", "40.0", "1"]]

    def test_truncates_extra_columns(self):
        text = "```csv\n5,40.0,1,0,0\n```"
        rows = extract_rows(text, n_cols=3)
        assert rows == [["5", "40.0", "1"]]


class TestValidateRows:
    def test_valid_row_accepted(self, stats):
        raw = [["5", "40.0", "1"]]
        good = validate_rows(raw, COLUMNS, stats, "target", 1)
        assert len(good) == 1
        assert good[0]["target"] == 1

    def test_out_of_range_row_rejected(self, stats):
        raw = [["9999", "40.0", "1"]]
        good = validate_rows(raw, COLUMNS, stats, "target", 1)
        assert len(good) == 0

    def test_non_numeric_row_rejected(self, stats):
        raw = [["abc", "40.0", "1"]]
        good = validate_rows(raw, COLUMNS, stats, "target", 1)
        assert len(good) == 0

    def test_wrong_length_row_rejected(self, stats):
        raw = [["5", "40.0"]]
        good = validate_rows(raw, COLUMNS, stats, "target", 1)
        assert len(good) == 0


class TestPythonFallbacks:
    def test_smote_fallback_generates_requested_rows(self, minority_rows, stats):
        rows = python_smote(minority_rows, COLUMNS, stats, "target", 1, n_rows=10, seed=1)
        assert len(rows) == 10
        assert all(r["target"] == 1 for r in rows)

    def test_gaussian_fallback_generates_requested_rows(self, minority_rows, stats):
        rows = python_gaussian_noise(minority_rows, COLUMNS, stats, "target", 1, n_rows=10, seed=1)
        assert len(rows) == 10
        assert all(r["target"] == 1 for r in rows)

    def test_smote_fallback_stays_in_range(self, minority_rows, stats):
        rows = python_smote(minority_rows, COLUMNS, stats, "target", 1, n_rows=20, seed=2)
        for r in rows:
            assert stats["income_ord"]["min"] <= r["income_ord"] <= stats["income_ord"]["max"]


class TestOllamaReachability:
    def test_unreachable_host_returns_false(self):
        assert is_ollama_reachable("http://localhost:1", timeout=0.5) is False
