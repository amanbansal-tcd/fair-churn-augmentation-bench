"""
model_comparison.py
====================
Cross-technique model comparison: trains Logistic Regression, Random
Forest, XGBoost and LightGBM on every available augmented-dataset variant
and evaluates all of them against the single held-out test set, producing
outputs/reports/model_comparison_report.csv.

Ported and adapted from the Classification project's run_experiment.py /
build_combined_datasets.py / evaluate.py (the "judge-and-filter" design),
merged into this project's dataset registry (outputs/augmented_datasets/)
and restricted to LR/RF/XGBoost/LightGBM (the Claude-few-shot and
hand-written "logic" classifiers from that project are explicitly out of
scope here).

Dataset registry
-----------------
Every technique writes (or is expected to write) two files into
outputs/augmented_datasets/:
  train_<name>_combined.csv   - original + that technique's synthetic rows
  train_<name>_synthetic.csv  - synthetic rows only
plus the always-present train_original.csv and test.csv.

`name` is one of: smote, adasyn, gaussian, ctgan, hnsba, mistral, phi.
Only variants whose combined CSV actually exists on disk are evaluated -
this lets `compare` run meaningfully even if CTGAN/HNSBA/LLM stages were
skipped.

Two additional "mega-variants" are built at run time from whichever
per-technique synthetic files exist:
  combined_all_sources - original + every available technique's synthetic
                         rows pooled together.
  combined_filtered    - original + only the synthetic sources that did
                         NOT underperform the original baseline (on BOTH
                         balanced_accuracy and F1) under a judge model
                         (the model with the highest mean F1 across all
                         phase-1 runs) - mirrors the Classification
                         project's two-phase selection process.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import AUG_DIR, REPORT_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

SYNTHETIC_NAMES = ["smote", "adasyn", "gaussian", "ctgan", "hnsba", "mistral", "phi"]

MODEL_FACTORIES = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForestClassifier": lambda: RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
    ),
    "XGBoostClassifier": lambda: XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
    ),
    "LightGBMClassifier": lambda: LGBMClassifier(
        n_estimators=300, max_depth=-1, learning_rate=0.1,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    ),
}


@dataclass
class Metrics:
    """Full binary-classification metric set, including confusion-matrix-derived rates."""
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "roc_auc": self.roc_auc, "specificity": self.specificity,
            "FPR": self.false_positive_rate, "FNR": self.false_negative_rate,
            "TN": self.true_negatives, "FP": self.false_positives,
            "FN": self.false_negatives, "TP": self.true_positives,
        }


def score_predictions(y_true, y_pred, y_proba) -> Metrics:
    """
    Compute accuracy/precision/recall/F1/ROC-AUC plus confusion-matrix
    rates (specificity, FPR, FNR). zero_division=0 avoids sklearn warnings
    when a model predicts a single class (can happen on tiny synthetic
    variants like the 300-row Mistral/Phi outputs).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        specificity=specificity, false_positive_rate=fpr, false_negative_rate=fnr,
        true_negatives=int(tn), false_positives=int(fp),
        false_negatives=int(fn), true_positives=int(tp),
    )


def _load_xy(path: Path):
    """Load a train/test CSV of features + trailing `target` column."""
    df = pd.read_csv(path)
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def discover_dataset_registry(aug_dir: Path = AUG_DIR) -> dict:
    """
    Scan AUG_DIR for available dataset variants.

    Returns
    -------
    dict
        {dataset_name: combined_csv_path}, always including "original" if
        present, plus one entry per synthetic technique whose combined CSV
        exists on disk.
    """
    registry = {}
    orig = aug_dir / "train_original.csv"
    if orig.exists():
        registry["original"] = orig
    for name in SYNTHETIC_NAMES:
        p = aug_dir / f"train_{name}_combined.csv"
        if p.exists():
            registry[f"combined_{name}"] = p
    return registry


def build_mega_variants(aug_dir: Path, judge_results: pd.DataFrame = None, judge_model: str = None) -> dict:
    """
    Build combined_all_sources (and, if judge info is supplied,
    combined_filtered) from whichever per-technique synthetic-only CSVs
    exist on disk, writing them into aug_dir and returning their paths.
    """
    orig_path = aug_dir / "train_original.csv"
    if not orig_path.exists():
        return {}

    available_synth = {
        name: aug_dir / f"train_{name}_synthetic.csv"
        for name in SYNTHETIC_NAMES
        if (aug_dir / f"train_{name}_synthetic.csv").exists()
    }

    out = {}

    # combined_all_sources: original + every available synthetic source
    frames = [pd.read_csv(orig_path)] + [pd.read_csv(p) for p in available_synth.values()]
    all_sources_path = aug_dir / "train_all_sources_combined.csv"
    pd.concat(frames, ignore_index=True).to_csv(all_sources_path, index=False)
    out["combined_all_sources"] = all_sources_path
    logger.info("Built combined_all_sources (%d sources)", len(available_synth))

    # combined_filtered: drop sources that underperform the judge model's
    # original baseline on BOTH balanced_accuracy and f1.
    if judge_results is not None and judge_model is not None and not judge_results.empty:
        baseline_rows = judge_results[
            (judge_results["dataset"] == "original") & (judge_results["model"] == judge_model)
        ]
        if not baseline_rows.empty:
            baseline = baseline_rows.iloc[0]
            dropped = []
            for name in available_synth:
                row = judge_results[
                    (judge_results["dataset"] == f"combined_{name}") & (judge_results["model"] == judge_model)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                if row["balanced_accuracy"] < baseline["balanced_accuracy"] and row["f1"] < baseline["f1"]:
                    dropped.append(name)
            kept = [n for n in available_synth if n not in dropped]
            frames = [pd.read_csv(orig_path)] + [pd.read_csv(available_synth[n]) for n in kept]
            filtered_path = aug_dir / "train_filtered_combined.csv"
            pd.concat(frames, ignore_index=True).to_csv(filtered_path, index=False)
            out["combined_filtered"] = filtered_path
            logger.info("Built combined_filtered (kept=%s, dropped=%s)", kept, dropped)

    return out


def run_one(dataset_name: str, dataset_path: Path, X_test, y_test) -> list:
    """Train every model in MODEL_FACTORIES on one dataset, evaluate on the shared test set."""
    X_train_raw, y_train = _load_xy(dataset_path)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test)

    rows = []
    for model_name, make_model in MODEL_FACTORIES.items():
        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics = score_predictions(y_test, y_pred, y_proba)

        row = {
            "dataset": dataset_name, "model": model_name,
            "n_train_rows": len(X_train_raw),
            **metrics.as_dict(),
        }
        rows.append(row)
        logger.info(
            "%-25s %-25s acc=%.4f bal_acc=%.4f f1=%.4f auc=%.4f",
            dataset_name, model_name, metrics.accuracy, metrics.balanced_accuracy,
            metrics.f1, metrics.roc_auc,
        )
    return rows


def run_model_comparison(aug_dir: Path = None, report_dir: Path = None) -> pd.DataFrame:
    """
    Full model-comparison pipeline: discover available dataset variants,
    train LR/RF/XGBoost/LightGBM on each, build the all-sources /
    filtered mega-variants using a judge model, and write
    outputs/reports/model_comparison_report.csv.

    Returns
    -------
    pd.DataFrame
        One row per (dataset, model) with the full metric set plus
        n_train_rows and new_rows.
    """
    aug_dir = Path(aug_dir) if aug_dir else AUG_DIR
    report_dir = Path(report_dir) if report_dir else REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    test_path = aug_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Held-out test set not found: {test_path}. Run the augmentation stage first.")
    X_test, y_test = _load_xy(test_path)

    registry = discover_dataset_registry(aug_dir)
    if "original" not in registry:
        raise FileNotFoundError(f"train_original.csv not found in {aug_dir}.")

    logger.info("Phase 1: evaluating %d dataset variants", len(registry))
    phase1_rows = []
    for name, path in registry.items():
        phase1_rows.extend(run_one(name, path, X_test, y_test))
    phase1_df = pd.DataFrame(phase1_rows)

    judge_model = phase1_df.groupby("model")["f1"].mean().idxmax()
    logger.info("Judge model (highest mean F1 in phase 1): %s", judge_model)

    mega = build_mega_variants(aug_dir, judge_results=phase1_df, judge_model=judge_model)
    phase2_rows = []
    for name, path in mega.items():
        phase2_rows.extend(run_one(name, path, X_test, y_test))
    phase2_df = pd.DataFrame(phase2_rows) if phase2_rows else pd.DataFrame()

    results = pd.concat([phase1_df, phase2_df], ignore_index=True) if not phase2_df.empty else phase1_df

    original_rows = results.loc[results["dataset"] == "original", "n_train_rows"].iloc[0]
    results["new_rows"] = results["n_train_rows"] - original_rows

    output_cols = [
        "dataset", "model", "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "roc_auc", "specificity", "FPR", "FNR", "TP", "FP", "TN", "FN",
        "n_train_rows", "new_rows",
    ]
    report_df = results[output_cols]
    report_df.to_csv(report_dir / "model_comparison_report.csv", index=False)
    logger.info("Saved %s", report_dir / "model_comparison_report.csv")

    return report_df
