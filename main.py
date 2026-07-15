#!/usr/bin/env python3
"""
main.py
=======
Single entry-point for the churn-prediction augmentation benchmark
pipeline.

Usage
-----
    python main.py --stage all --no-ctgan --no-llm   # fast, offline path
    python main.py --stage eda
    python main.py --stage baseline
    python main.py --stage fairness
    python main.py --stage mitigation
    python main.py --stage augmentation --no-ctgan
    python main.py --stage llm-augmentation
    python main.py --stage compare
    python main.py --stage report

Stages
------
  eda               - EDA + visualisations
  baseline          - LR/RF/XGBoost/LightGBM/CatBoost baseline + SHAP
  fairness          - Fairness audit across sensitive attributes
  mitigation        - M0-M15 feature-removal + R1-R6 rebalancing + Pareto
  augmentation      - SMOTE / ADASYN / Gaussian / CTGAN / HNSBA (Claude)
  llm-augmentation  - Mistral / Phi augmentation via local Ollama
  compare           - Train LR/RF/XGBoost/LightGBM on every dataset variant
  report            - Generate outputs/reports/report.docx from result CSVs
  all               - eda -> baseline -> fairness -> mitigation ->
                       augmentation -> llm-augmentation -> compare -> report
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_FEATURES, TEST_SIZE, RANDOM_SEED,
    FIG_DIR, MODEL_DIR, REPORT_DIR, AUG_DIR,
)
from src.data_loader import load_data
from src.preprocessing import build_features
from src.eda import run_eda
from src.models import build_models, train_all, cross_validate_model, save_model
from src.evaluation import plot_diagnostics, metrics_dataframe, evaluate_model
from src.feature_importance import plot_feature_importance, plot_shap
from src.fairness import run_fairness_audit, plot_fairness_dashboard
from src.bias_mitigation import run_mitigation_experiments
from src.rebalancing import run_rebalancing_experiments
from src.augmentation.classical import (
    run_augmentation, export_augmented_csvs,
    run_augmentation_evaluation, plot_augmentation_figures,
)
from src.pareto import build_master_table, plot_pareto, plot_heatmap
from src.final_model import get_final_features, train_final_model, plot_final_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run(stage: str = "all", run_ctgan: bool = True, run_llm: bool = True) -> None:
    t0 = time.time()

    for d in [FIG_DIR, MODEL_DIR, REPORT_DIR, AUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    stages_needing_data = {"eda", "baseline", "fairness", "mitigation", "augmentation", "all"}
    if stage in stages_needing_data:
        logger.info("Loading dataset...")
        df_raw = load_data()
        X, y, df_clean, encoders = build_features(df_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
        )
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(
            "Split: train=%d  test=%d  pos_rate=%.4f  scale_pos_weight=%.2f",
            len(X_train), len(X_test), y_train.mean(), spw,
        )

    if stage in ("eda", "all"):
        _banner("EDA")
        run_eda(df_raw, df_clean, FIG_DIR)

    if stage in ("baseline", "all"):
        _banner("BASELINE MODELS")
        models = build_models(spw)
        results = train_all(models, X_train, X_test, y_train, y_test)

        logger.info("5-fold CV on LightGBM:")
        cross_validate_model(build_models(spw)["LightGBM"], X_train, y_train)

        plot_diagnostics(results, y_test, FIG_DIR)
        plot_feature_importance(models, ALL_FEATURES, FIG_DIR)
        plot_shap(models["LightGBM"], X_test, ALL_FEATURES, FIG_DIR)

        for name, model in models.items():
            save_model(model, name, MODEL_DIR)

        _print_table("BASELINE PERFORMANCE", metrics_dataframe(results))

    if stage in ("fairness", "all"):
        _banner("FAIRNESS AUDIT")
        cat_model = _load_or_train("CatBoost", build_models(spw), X_train, y_train)
        yp = cat_model.predict(X_test)
        ypr = cat_model.predict_proba(X_test)[:, 1]
        audit = run_fairness_audit(y_test.values, yp, ypr, df_clean, X_test.index)
        plot_fairness_dashboard(audit, FIG_DIR)

    if stage in ("mitigation", "all"):
        _banner("BIAS MITIGATION (M0-M15 + R1-R6)")
        mit_df = run_mitigation_experiments(df_clean, X_train, X_test, y_train, y_test)
        reb_df = run_rebalancing_experiments(df_clean, X_train, X_test, y_train, y_test)

        baseline_results = _rebuild_baseline(build_models(spw), X_train, X_test, y_train, y_test)
        try:
            cat_m = _load_or_train("CatBoost", build_models(spw), X_train, y_train)
            yp2 = cat_m.predict(X_test)
            ypr2 = cat_m.predict_proba(X_test)[:, 1]
            fairness = run_fairness_audit(y_test.values, yp2, ypr2, df_clean, X_test.index)
        except Exception:
            fairness = None

        master = build_master_table(baseline_results, mit_df, reb_df, fairness)
        plot_pareto(master, FIG_DIR)
        plot_heatmap(master, FIG_DIR)

    if stage in ("augmentation", "all"):
        _banner("DATA AUGMENTATION (SMOTE / ADASYN / Gaussian / CTGAN)")
        if not run_ctgan:
            logger.info("CTGAN disabled (--no-ctgan flag set).")

        aug_dict = run_augmentation(X_train, y_train, run_ctgan=run_ctgan)
        export_augmented_csvs(aug_dict, X_test, y_test, AUG_DIR)
        aug_results = run_augmentation_evaluation(aug_dict, X_test, y_test)
        plot_augmentation_figures(aug_dict, aug_results, y_test, FIG_DIR)

        _run_hnsba_stage(X_train, y_train, X_test, y_test)

        print("\n" + "=" * 78)
        print("  AUGMENTATION PERFORMANCE COMPARISON (same test set)")
        print("=" * 78)
        orig_auc = aug_results.get("original", {}).get("roc_auc", 0)
        for name, res in aug_results.items():
            d = res["roc_auc"] - orig_auc
            print(f"  {name:<22}  AUC={res['roc_auc']:.4f} ({'+' if d>=0 else ''}{d:.4f})  "
                  f"F1={res['f1']:.4f}  Recall={res['recall']:.4f}")
        print("=" * 78 + "\n")

        print("Augmented datasets exported to outputs/augmented_datasets/:")
        for f in sorted(AUG_DIR.glob("*.csv")):
            n_rows = len(pd.read_csv(f))
            print(f"  {f.name:<40}  {n_rows:>7,} rows")
        print()

    if stage == "llm-augmentation":
        _run_llm_stage(run_llm)

    if stage == "all":
        _run_llm_stage(run_llm)

    if stage in ("compare", "all"):
        _banner("MODEL COMPARISON ACROSS DATASET VARIANTS")
        from src.model_comparison import run_model_comparison
        try:
            report_df = run_model_comparison()
            _print_table("MODEL COMPARISON (best per dataset)",
                          report_df.sort_values("f1", ascending=False).groupby("dataset").first().reset_index())
        except FileNotFoundError as exc:
            logger.warning("Skipping compare stage: %s", exc)

    if stage in ("report", "all"):
        _banner("REPORT GENERATION")
        from src.report import generate_report
        path = generate_report()
        logger.info("Report written to %s", path)

    elapsed = time.time() - t0
    n_figs = len(list(FIG_DIR.glob("*.png")))
    n_mods = len(list(MODEL_DIR.glob("*.pkl")))
    n_rpts = len(list(REPORT_DIR.glob("*.csv")))
    n_csvs = len(list(AUG_DIR.glob("*.csv")))
    logger.info(
        "Done in %.0fs  |  Figures: %d  |  Models: %d  |  Reports: %d  |  Augmented CSVs: %d",
        elapsed, n_figs, n_mods, n_rpts, n_csvs,
    )


def _run_hnsba_stage(X_train, y_train, X_test, y_test) -> None:
    """Run HNSBA (Claude-based) augmentation and export its combined + synthetic-only CSVs."""
    from src.augmentation.hnsba import run_hnsba_augmentation

    result = run_hnsba_augmentation(X_train, y_train)
    if result is None:
        return

    AUG_DIR.mkdir(parents=True, exist_ok=True)

    combined_df = result["X_combined"].copy()
    combined_df["target"] = result["y_combined"].values
    combined_df.to_csv(AUG_DIR / "train_hnsba_combined.csv", index=False)

    synth_df = result["X_synthetic"].copy()
    synth_df["target"] = result["y_synthetic"].values
    synth_df.to_csv(AUG_DIR / "train_hnsba_synthetic.csv", index=False)

    logger.info(
        "HNSBA datasets exported: combined=%d rows, synthetic=%d rows.",
        len(combined_df), len(synth_df),
    )


def _run_llm_stage(run_llm: bool) -> None:
    """Run the Ollama-based (Mistral/Phi) augmentation stage, skipping gracefully if unreachable."""
    if not run_llm:
        logger.info("LLM augmentation disabled (--no-llm flag set).")
        return

    from src.augmentation.llm_aug import run_llm_augmentation, is_ollama_reachable, DEFAULT_OLLAMA_URL

    if not is_ollama_reachable(DEFAULT_OLLAMA_URL):
        print(
            f"Ollama server not reachable at {DEFAULT_OLLAMA_URL} - skipping LLM "
            "augmentation stage. Start Ollama and re-run with `main.py llm-augmentation` "
            "to generate Mistral/Phi synthetic data."
        )
        return

    _banner("LLM AUGMENTATION (Mistral / Phi via Ollama)")
    input_csv = AUG_DIR / "train_original.csv"
    if not input_csv.exists():
        logger.warning("train_original.csv not found; run the augmentation stage first.")
        return

    out_root = Path("outputs") / "llm_augmentation"
    log_root = Path("outputs") / "llm_augmentation_logs"
    results = run_llm_augmentation(input_csv, out_root, log_root)

    for name, info in results.items():
        combined_src = info["combined_path"]
        dest = AUG_DIR / f"train_{name}_combined.csv"
        pd.read_csv(combined_src).to_csv(dest, index=False)
        orig_df = pd.read_csv(input_csv)
        combined_df = pd.read_csv(combined_src)
        synth_df = combined_df.iloc[len(orig_df):]
        synth_df.to_csv(AUG_DIR / f"train_{name}_synthetic.csv", index=False)
        logger.info("%s: %d synthetic rows exported.", name, info["n_rows"])


def _banner(title: str) -> None:
    logger.info("=" * 55)
    logger.info("  STAGE: %s", title)
    logger.info("=" * 55)


def _print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


def _load_or_train(name: str, models: dict, X_train, y_train):
    path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
    if path.exists():
        import joblib
        return joblib.load(path)
    m = models[name]
    m.fit(X_train, y_train)
    return m


def _rebuild_baseline(models: dict, X_train, X_test, y_train, y_test) -> dict:
    results = {}
    for name in models:
        m = _load_or_train(name, models, X_train, y_train)
        yp = m.predict(X_test)
        ypr = m.predict_proba(X_test)[:, 1]
        results[name] = evaluate_model(y_test, yp, ypr)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Churn augmentation benchmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", default="all",
        choices=["eda", "baseline", "fairness", "mitigation", "augmentation",
                 "llm-augmentation", "compare", "report", "all"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument("--no-ctgan", action="store_true", help="Skip CTGAN augmentation")
    parser.add_argument("--no-llm", action="store_true", help="Skip Mistral/Phi (Ollama) augmentation")
    args = parser.parse_args()
    run(stage=args.stage, run_ctgan=not args.no_ctgan, run_llm=not args.no_llm)


if __name__ == "__main__":
    main()
