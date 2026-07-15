# Fair Churn Augmentation Bench

A research pipeline comparing data augmentation strategies for imbalanced
subscriber-churn prediction, with an integrated fairness audit and bias
mitigation study.

The dataset is a newspaper subscription dataset with a ~4.2:1 class
imbalance (NO:YES) and four demographic attributes of interest:
Ethnicity, Language, Home Ownership, and Age range. The pipeline trains
several classifiers, audits them for group fairness, mitigates the bias
found, and then benchmarks seven augmentation techniques - classical
resampling (SMOTE, ADASYN), statistical perturbation (Gaussian noise), a
generative model (CTGAN), and three LLM-based generators (a
schema-constrained Claude procedure referred to as HNSBA, plus Mistral
and Phi via a local Ollama server) - against a single held-out test set.

## Repository Structure

```
fair-churn-augmentation-bench/
  main.py                     CLI entry point (see Stages below)
  src/
    config.py                 paths, constants, hyperparameters
    data_loader.py             CSV loading + schema validation
    preprocessing.py          feature engineering / encoding
    eda.py                     exploratory data analysis figures
    models.py                  baseline model factories + training
    evaluation.py              metrics + diagnostic plots
    feature_importance.py     SHAP-based feature importance
    fairness.py                 8-metric group fairness audit
    bias_mitigation.py        M0-M15 feature-removal experiments
    rebalancing.py            R1-R6 reweighting experiments
    pareto.py                  fairness/performance Pareto frontier
    final_model.py             M12 (recommended debiased model)
    model_comparison.py       trains LR/RF/XGBoost/LightGBM on every
                               dataset variant and reports metrics
    report.py                  generates outputs/reports/report.docx
    augmentation/
      classical.py             SMOTE, ADASYN, Gaussian noise
      ctgan_aug.py              CTGAN
      hnsba.py                  Claude-based schema-bound augmentation
      llm_aug.py                Mistral/Phi augmentation via Ollama
  tests/                       pytest suite (unit tests, no network calls)
  data/                        place NewspaperChurn.csv here (gitignored)
  outputs/
    figures/                   generated PNG figures
    models/                    saved model artifacts (.pkl)
    reports/                   CSV result tables + report.docx
    augmented_datasets/        exported train/test CSV variants
```

## Setup

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

Place the dataset at `data/NewspaperChurn.csv` (see `data/README.md`).

Optional, for LLM-based augmentation:
- `ANTHROPIC_API_KEY` environment variable, for the HNSBA stage.
- A local Ollama server (`http://localhost:11434`) with `mistral:latest`
  and `phi:latest` pulled, for the Mistral/Phi stage.

## Usage

```bash
# Fast, fully offline path (skips CTGAN and the LLM stages)
python main.py --stage all --no-ctgan --no-llm

# Individual stages
python main.py --stage eda
python main.py --stage baseline
python main.py --stage fairness
python main.py --stage mitigation
python main.py --stage augmentation --no-ctgan
python main.py --stage llm-augmentation
python main.py --stage compare
python main.py --stage report
```

`llm-augmentation` checks whether the Ollama server is reachable before
attempting anything; if it isn't, the stage prints a clear message and
exits cleanly rather than failing the whole run. `augmentation` runs
CTGAN and HNSBA in addition to the classical techniques unless
`--no-ctgan` is passed or `ANTHROPIC_API_KEY` is unset, respectively.

## Results

**Best baseline classifier:** LightGBM, the strongest performer among
Logistic Regression, Random Forest, XGBoost, and LightGBM on the
unaugmented training data.

**Best augmentation result:** a Mistral-augmented training set paired
with LightGBM, reaching an F1 of approximately 0.567 - the strongest
result observed among the augmentation techniques evaluated.

**Fairness audit:** all four sensitive attributes examined (Ethnicity,
Language, Home Ownership, Age range) fail the EEOC four-fifths rule
(Disparate Impact Ratio below 0.80) under the full-feature baseline
model.

**Recommended debiased model:** M12 (LightGBM with Ethnicity, Language,
and Home Ownership removed from the feature set), which substantially
improves fairness metrics at a modest cost to raw predictive
performance.

**Known limitation:** minority-class (churner) detection plateaus at
roughly 50% recall across every technique and model evaluated. This
ceiling is attributed to the limits of the available feature signal
rather than to modelling or augmentation choices - no technique tested
here moved recall meaningfully past this point.

All of the numbers above are regenerated from the CSVs in
`outputs/reports/` on every `report` stage run - see
`outputs/reports/model_comparison_report.csv`,
`outputs/reports/fairness_summary.csv`, and
`outputs/reports/mitigation_results.csv` for the underlying figures from
your own run.

## Tests

```bash
pytest tests/ -v
```

The test suite (unit tests, no network calls) covers preprocessing,
evaluation, fairness metrics, the classical augmentation techniques, and
the validation/repair logic in the HNSBA and Ollama-based augmentation
modules.

## License

MIT - see `LICENSE`.
