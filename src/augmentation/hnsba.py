"""
hnsba.py
========
HNSBA - Hierarchical, Numerically-Schema-Bound Augmentation.

An LLM-based (Anthropic Claude) minority-class augmentation technique that
constrains generation to the existing schema and to per-feature value
ranges / marginal distributions observed in the real minority class,
producing rows in small, independently verifiable batches rather than one
large freeform request.

Three components (mirrors the ollama-augmentation validation style):
  1. Profiler   - compute_marginal_profile(): per-feature min/max/mean/
                  stdev and categorical/integer-code frequency tables from
                  the real minority-class rows.
  2. Generator  - generate_batches(): a batched prompting loop that asks
                  Claude for N rows as strict JSON constrained to the
                  profiled ranges, with retry-on-malformed-JSON.
  3. Validator  - validate_and_repair(): clips numeric values to the
                  observed range, coerces categorical/integer-encoded
                  values to the observed value set (nearest valid code),
                  and rounds integer columns.

Requires the ANTHROPIC_API_KEY environment variable. If it is not set,
run_hnsba_augmentation() logs a clear error and returns None so that
main.py can skip this stage gracefully instead of crashing.
"""

import json
import logging
import os
import time

import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, FEATURE_RANGES, RANDOM_SEED

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"
MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 60
DEFAULT_N_SYNTH = 3_570


# 1. Profiler

def compute_marginal_profile(X_minority: pd.DataFrame) -> dict:
    """
    Compute per-feature statistics from real minority-class rows.

    For continuous ("float") features this is min/max/mean/stdev, used to
    bound and shape the prompt's requested ranges. For integer-encoded
    ("int") features — which in this schema are really categorical codes
    (e.g. Ethnicity_enc) — this additionally records the observed set of
    valid codes and their relative frequency, so the validator can snap
    any out-of-vocabulary code the model invents back to a real one.

    Parameters
    ----------
    X_minority : pd.DataFrame
        Feature rows for the minority class only.

    Returns
    -------
    dict
        {feature_name: {"type": "int"|"float", "min":, "max":, "mean":,
                         "stdev":, "categories": [...] (int features only)}}
    """
    profile = {}
    for col in X_minority.columns:
        cfg = FEATURE_RANGES.get(col, {})
        vals = X_minority[col].astype(float)
        entry = {
            "type": cfg.get("type", "float"),
            "min": float(cfg.get("min", vals.min())),
            "max": float(cfg.get("max", vals.max())),
            "mean": round(float(vals.mean()), 4),
            "stdev": round(float(vals.std() or 0.0), 4),
        }
        if entry["type"] == "int":
            counts = vals.round().astype(int).value_counts(normalize=True)
            entry["categories"] = sorted(counts.index.tolist())
            entry["category_freq"] = {int(k): round(float(v), 4) for k, v in counts.items()}
        profile[col] = entry
    return profile


# 2. Generator

def _build_batch_prompt(profile: dict, columns: list, batch_size: int) -> str:
    """Build the strict-JSON batch-generation prompt from the marginal profile."""
    col_lines = []
    for col in columns:
        p = profile[col]
        if p["type"] == "int":
            cats = p.get("categories", [])
            cats_str = ", ".join(str(c) for c in cats[:40])
            col_lines.append(
                f'  "{col}": integer code, must be one of [{cats_str}] '
                f"(observed range [{int(p['min'])}, {int(p['max'])}])"
            )
        else:
            col_lines.append(
                f'  "{col}": float, range [{p["min"]}, {p["max"]}], '
                f"mean={p['mean']}, stdev={p['stdev']}"
            )

    schema_cols = ", ".join(f'"{c}"' for c in columns)
    return (
        "You are a synthetic tabular data generator. You must never invent "
        "new columns, and every value you output must be a valid, in-range "
        "value for its column as defined below.\n\n"
        f"Generate exactly {batch_size} synthetic rows for the minority class "
        "of a churn-prediction dataset, matching this schema:\n\n"
        + "\n".join(col_lines)
        + "\n\nOutput STRICT JSON only: a list of objects, each with exactly "
        f"these keys in this order: [{schema_cols}]. No prose, no markdown "
        "fences, no explanations - just the JSON array."
    )


def generate_batches(
    profile: dict,
    columns: list,
    n_synth: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    client=None,
) -> list:
    """
    Batched Claude prompting loop: repeatedly ask for `batch_size` rows as
    strict JSON until `n_synth` rows are collected. Malformed JSON
    responses are retried up to MAX_RETRIES times per batch before that
    batch is skipped (verifiable per-batch design: a bad batch never
    corrupts previously collected rows).

    Parameters
    ----------
    profile : dict
        Output of compute_marginal_profile().
    columns : list
        Feature column order (matches ALL_FEATURES).
    n_synth : int
        Total synthetic rows to generate.
    batch_size : int
        Rows requested per API call.
    client : anthropic.Anthropic | None
        Injectable client, primarily for testing; created lazily otherwise.

    Returns
    -------
    list[dict]
        Raw (unvalidated) row dicts collected across all batches.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    rows: list = []
    while len(rows) < n_synth:
        remaining = n_synth - len(rows)
        this_batch = min(batch_size, remaining)
        prompt = _build_batch_prompt(profile, columns, this_batch)

        parsed = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                # Strip accidental markdown fences before parsing.
                if text.startswith("```"):
                    text = text.strip("`")
                    text = text[text.find("["):]
                parsed = json.loads(text)
                break
            except (json.JSONDecodeError, IndexError, AttributeError) as exc:
                logger.warning("HNSBA batch parse failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                time.sleep(2 * attempt)
            except Exception as exc:
                logger.warning("HNSBA API call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                time.sleep(2 * attempt)

        if parsed is None:
            logger.error("HNSBA batch permanently failed after %d retries; skipping batch.", MAX_RETRIES)
            continue

        if isinstance(parsed, list):
            rows.extend(parsed)
        logger.info("HNSBA: collected %d/%d rows", len(rows), n_synth)

    return rows[:n_synth]


# 3. Validator / repair

def validate_and_repair(raw_rows: list, profile: dict, columns: list) -> pd.DataFrame:
    """
    Repair raw LLM-generated rows against the observed schema:
      - missing/non-numeric values are dropped (row rejected);
      - numeric values are clipped to the observed [min, max];
      - integer-coded (categorical) columns are snapped to the nearest
        observed category code if the model emits an out-of-vocabulary
        value;
      - integer columns are rounded to whole numbers.

    Parameters
    ----------
    raw_rows : list[dict]
        Unvalidated rows from generate_batches().
    profile : dict
        Output of compute_marginal_profile().
    columns : list
        Expected feature column order.

    Returns
    -------
    pd.DataFrame
        Repaired rows, one per valid input row (invalid rows dropped).
    """
    repaired = []
    for row in raw_rows:
        try:
            clean = {}
            for col in columns:
                val = float(row[col])
                p = profile[col]
                val = min(max(val, p["min"]), p["max"])
                if p["type"] == "int":
                    cats = p.get("categories", [])
                    ival = int(round(val))
                    if cats and ival not in cats:
                        ival = min(cats, key=lambda c: abs(c - ival))
                    clean[col] = ival
                else:
                    clean[col] = val
            repaired.append(clean)
        except (KeyError, ValueError, TypeError):
            continue

    return pd.DataFrame(repaired, columns=columns)


# Public entry point

def run_hnsba_augmentation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_synth: int = DEFAULT_N_SYNTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Run the full HNSBA pipeline: profile the real minority class, generate
    synthetic rows via Claude in verifiable batches, repair/validate them,
    and return a combined (original + synthetic) training set.

    Returns None (after logging a clear error) if ANTHROPIC_API_KEY is not
    set, so main.py can skip this stage without crashing.

    Returns
    -------
    dict | None
        {"X_combined", "y_combined", "X_synthetic", "y_synthetic"} - the
        combined set is shuffled with RANDOM_SEED; the synthetic-only
        frames are kept unshuffled so callers can export both a combined
        and a synthetic-only CSV without re-deriving the split. Returns
        None if ANTHROPIC_API_KEY is not set.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set - skipping HNSBA augmentation. "
            "Set the environment variable and re-run to generate Claude-based synthetic data."
        )
        return None

    columns = list(X_train.columns)
    X_min = X_train[y_train == 1]
    profile = compute_marginal_profile(X_min)

    logger.info("HNSBA: generating %d synthetic minority rows...", n_synth)
    raw_rows = generate_batches(profile, columns, n_synth, batch_size)
    X_synth = validate_and_repair(raw_rows, profile, columns)
    logger.info("HNSBA: %d/%d rows survived validation", len(X_synth), n_synth)

    y_synth = pd.Series(np.ones(len(X_synth), dtype=int), name=y_train.name)

    X_combined = pd.concat([X_train.reset_index(drop=True), X_synth.reset_index(drop=True)], ignore_index=True)
    y_combined = pd.concat([y_train.reset_index(drop=True), y_synth], ignore_index=True)

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_combined))
    return {
        "X_combined": X_combined.iloc[perm].reset_index(drop=True),
        "y_combined": y_combined.iloc[perm].reset_index(drop=True),
        "X_synthetic": X_synth.reset_index(drop=True),
        "y_synthetic": y_synth.reset_index(drop=True),
    }
