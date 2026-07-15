"""
llm_aug.py
==========
Local-LLM (Ollama) minority-class augmentation, refactored from the
standalone ollama-augmentation/augment.py script into functions callable
from main.py's `llm-augmentation` stage.

Only mistral:latest and phi:latest are supported here (deepseek-r1 is
explicitly out of scope for this project — it is a slow reasoning model
and was only a secondary path in the original script).

How it works
------------
1. Each model is first asked (Step 1) to pick and explain an augmentation
   technique for the observed minority-class statistics (min/max/mean/std
   per column) — this is a free-text explanation, not itself validated.
2. The model is then driven in a batched generation loop (Step 2), asking
   for `batch_size` synthetic CSV rows per call. Responses are parsed with
   `extract_rows` (fenced ```csv blocks, pipe-tables, or bare numeric
   lines) and validated with `validate_rows` (numeric, in-range, integer
   columns coerced).
3. If a model produces zero valid rows for `max_empty_batches` consecutive
   batches, a Python fallback (SMOTE-style linear interpolation or
   Gaussian noise, matching the technique family the model picked in
   Step 1) generates the remaining rows so the pipeline never stalls on
   an uncooperative small model.

Ollama call settings
---------------------
- Streaming (`stream: True`) so slow model responses never hit a single
  request timeout; a wall-clock cap per call still applies.
- `num_ctx=4096` is set explicitly in the request options to keep the
  model's context window bounded and avoid out-of-memory crashes on
  constrained hardware (this project's addition over the original
  script, which left num_ctx at the Ollama default).
- `call_with_retry` retries transient 500s and connection errors with
  backoff, up to MAX_RETRIES attempts.
"""

import csv
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from statistics import mean, stdev

import requests

logger = logging.getLogger(__name__)

# Model registry: name -> (ollama tag, batch_size, use_completion_prompt,
#                           python_fallback_technique, max_empty_batches, stream_timeout_s)
MODELS = {
    "phi":     ("phi:latest",     5, True,  "smote", 8, 600),
    "mistral": ("mistral:latest", 20, False, "smote", 8, 600),
}

DEFAULT_TARGET_COL = "target"
DEFAULT_MINORITY_CLASS = 1
DEFAULT_OLLAMA_URL = "http://localhost:11434"
MAX_RETRIES = 3
INTER_MODEL_DELAY = 20
NUM_CTX = 4096  # bounded context window to avoid OOM on constrained hardware


# Data utilities

def load_csv(path: Path):
    """Load a CSV file into (column_names, list_of_row_dicts)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = list(reader.fieldnames)
    return cols, rows


def compute_stats(rows: list, cols: list) -> dict:
    """Per-column min/max/mean/stdev and integer-ness, over numeric values only."""
    stats = {}
    for col in cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, KeyError):
                pass
        if vals:
            stats[col] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(mean(vals), 4),
                "stdev": round(stdev(vals) if len(vals) > 1 else 0.0, 4),
                "is_integer": all(v == int(v) for v in vals),
            }
    return stats


def sample_rows(rows: list, n: int = 6) -> list:
    """Evenly-spaced sample of n rows (used as few-shot examples in prompts)."""
    step = max(1, len(rows) // n)
    return rows[::step][:n]


# Ollama interface

def ollama_stream(ollama_url: str, model_tag: str, prompt: str, max_seconds: int = 600) -> str:
    """
    Stream a completion from the Ollama /api/generate endpoint.

    Streaming avoids single-request timeouts on slow models; a wall-clock
    cap (max_seconds) still bounds worst-case latency. num_ctx=4096 bounds
    memory use.
    """
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.7, "num_predict": 2048, "num_ctx": NUM_CTX},
    }
    parts = []
    t0 = time.time()
    with requests.post(f"{ollama_url}/api/generate", json=payload, stream=True, timeout=(60, None)) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if time.time() - t0 > max_seconds:
                logger.info("Ollama stream capped at %ds", max_seconds)
                break
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
                parts.append(data.get("response", ""))
                if data.get("done", False):
                    break
            except (json.JSONDecodeError, KeyError):
                pass
    return "".join(parts)


def call_with_retry(ollama_url: str, model_tag: str, prompt: str, retries: int = MAX_RETRIES, **kw) -> str:
    """Retry ollama_stream on transient HTTP 500s / connection errors, with backoff."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            return ollama_stream(ollama_url, model_tag, prompt, **kw)
        except requests.HTTPError as e:
            last = e
            code = e.response.status_code if e.response is not None else 0
            if code == 500 and attempt < retries:
                wait = 20 * attempt
                logger.warning("Ollama 500, retry %d/%d in %ds", attempt, retries, wait)
                time.sleep(wait)
            else:
                raise
        except requests.ConnectionError as e:
            last = e
            if attempt < retries:
                logger.warning("Ollama connection error, retry %d/%d", attempt, retries)
                time.sleep(10)
            else:
                raise
    raise last


# Prompts

def build_technique_prompt(columns: list, stats: dict) -> str:
    """Step-1 prompt: ask the model to pick and justify an augmentation technique."""
    lines = []
    for col in columns:
        if col not in stats:
            continue
        s = stats[col]
        kind = "integer" if s["is_integer"] else "float"
        lines.append(f"  {col}: {kind}, range [{s['min']}, {s['max']}], mean={s['mean']}, stdev={s['stdev']}")

    return (
        "You are a machine learning data augmentation expert.\n\n"
        "Dataset: tabular, numeric, binary classification.\n"
        "Problem: class imbalance between majority and minority rows.\n"
        "Goal: generate synthetic minority-class rows.\n\n"
        "Minority-class column statistics:\n"
        + "\n".join(lines)
        + "\n\nChoose ONE augmentation technique (e.g. Gaussian noise, linear "
        "interpolation, SMOTE-like sampling, conditional marginal sampling, etc.).\n"
        "Reply in 2-4 sentences: name the technique and explain why it fits this data.\n"
        "Do NOT generate any data rows yet.\n"
    )


def build_data_prompt(columns: list, stats: dict, samples: list, technique_name: str, batch_size: int) -> str:
    """Step-2 instruction-style data prompt for capable models (mistral)."""
    n = len(columns)
    col_lines = []
    for col in columns:
        if col not in stats:
            continue
        s = stats[col]
        kind = "integer" if s["is_integer"] else "float"
        col_lines.append(f"  col {columns.index(col)+1:02d} {col}: {kind} [{s['min']}, {s['max']}]")

    sample_lines = [",".join(str(r[c]) for c in columns) for r in samples]
    cols_str = ",".join(columns)

    return (
        f"Generate exactly {batch_size} synthetic CSV rows using {technique_name}.\n\n"
        f"There are exactly {n} columns in this order:\n"
        f"  {cols_str}\n\n"
        "Column types and ranges:\n"
        + "\n".join(col_lines)
        + "\n\nReal minority-class example rows (no header):\n"
        + "\n".join(sample_lines)
        + f"\n\nRULES - follow exactly:\n"
        f"1. Output ONLY {batch_size} data rows, nothing else.\n"
        f"2. Each row must have EXACTLY {n} comma-separated values.\n"
        f"3. Integer columns: no decimal point (write 7 not 7.0).\n"
        f"4. All values within the given [min, max] ranges.\n"
        f"5. Last column ({columns[-1]}) must always be 1.\n"
        f"6. No header row, no comments, no explanations.\n"
        f"7. Wrap all {batch_size} rows in one ```csv block.\n\n"
        "```csv\n"
    )


def build_completion_prompt(columns: list, samples: list) -> str:
    """Step-2 pattern-completion prompt for small models that ignore instructions (phi)."""
    header = ",".join(columns)
    sample_lines = [",".join(str(r[c]) for c in columns) for r in samples]
    return header + "\n" + "\n".join(sample_lines) + "\n"


# Parsing & validation

def strip_think_tags(text: str) -> str:
    """Remove any <think>...</think> reasoning blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_rows(text: str, n_cols: int) -> list:
    """
    Pull CSV data rows from raw model output text.

    Tries fenced ```csv blocks first, then pipe-tables, then bare numeric
    lines. Lines with more than n_cols values are truncated to n_cols
    (handles models that append stray extra columns); lines containing
    alphabetic characters (other than e/E for scientific notation) are
    rejected outright.
    """
    candidates = []

    for m in re.finditer(r"```(?:csv|text|)?\s*\n(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1).strip())

    table_lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            inner = line.strip("|").strip()
            if re.fullmatch(r"[\s\-\|:]+", inner):
                continue
            table_lines.append(",".join(p.strip() for p in inner.split("|")))
    if table_lines:
        candidates.append("\n".join(table_lines))

    if not candidates:
        bare = [l.strip() for l in text.splitlines() if re.match(r"^\s*-?\d", l) and l.count(",") >= n_cols - 2]
        if bare:
            candidates.append("\n".join(bare))

    rows, seen = [], set()
    for block in candidates:
        for line in block.splitlines():
            line = line.strip().rstrip(",")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if re.search(r"[a-df-wyzA-DF-WYZ]", line):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= n_cols and line not in seen:
                seen.add(line)
                rows.append(parts[:n_cols])

    return rows


def validate_rows(raw_rows: list, columns: list, stats: dict, target_col: str, target_class: int) -> list:
    """
    Coerce parsed row strings to typed dicts, rejecting rows with
    non-numeric or out-of-range values (range = observed [min, max]
    padded by one stdev). Integer columns are rounded.
    """
    good = []
    for parts in raw_rows:
        if len(parts) != len(columns):
            continue
        row, ok = {}, True
        for col, val in zip(columns, parts):
            try:
                fval = float(val)
            except ValueError:
                ok = False
                break
            if col == target_col:
                fval = float(target_class)
            elif col in stats:
                s = stats[col]
                lo = s["min"] - max(1.0, s["stdev"])
                hi = s["max"] + max(1.0, s["stdev"])
                if not (lo <= fval <= hi):
                    ok = False
                    break
                if s["is_integer"]:
                    fval = int(round(fval))
            row[col] = fval
        if ok:
            row[target_col] = target_class
            good.append(row)
    return good


# Python fallback generators (used when a model cannot produce valid CSV)

def python_smote(minority_rows, columns, stats, target_col, target_class, n_rows, seed=42):
    """Linear interpolation between random pairs of minority rows (SMOTE-style fallback)."""
    rng = random.Random(seed)

    float_rows = []
    for r in minority_rows:
        row, ok = {}, True
        for col in columns:
            try:
                row[col] = float(r[col])
            except (ValueError, KeyError):
                ok = False
                break
        if ok:
            float_rows.append(row)

    synthetic = []
    for _ in range(n_rows):
        r1, r2 = rng.sample(float_rows, 2)
        alpha = rng.random()
        row = {}
        for col in columns:
            if col == target_col:
                row[col] = target_class
                continue
            v = r1[col] + alpha * (r2[col] - r1[col])
            if col in stats and stats[col]["is_integer"]:
                v = int(round(v))
                v = max(int(stats[col]["min"]), min(int(stats[col]["max"]), v))
            else:
                v = round(v, 4)
            row[col] = v
        row[target_col] = target_class
        synthetic.append(row)

    return synthetic


def python_gaussian_noise(minority_rows, columns, stats, target_col, target_class, n_rows,
                           noise_fraction=0.05, seed=42):
    """Gaussian-noise perturbation of randomly sampled minority rows (fallback generator)."""
    rng = random.Random(seed)

    float_rows = []
    for r in minority_rows:
        row, ok = {}, True
        for col in columns:
            try:
                row[col] = float(r[col])
            except (ValueError, KeyError):
                ok = False
                break
        if ok:
            float_rows.append(row)

    synthetic = []
    for _ in range(n_rows):
        base = rng.choice(float_rows)
        row = {}
        for col in columns:
            if col == target_col:
                row[col] = target_class
                continue
            s = stats.get(col, {})
            sigma = s.get("stdev", 0) * noise_fraction
            u1, u2 = rng.random(), rng.random()
            noise = sigma * math.sqrt(-2 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
            v = base[col] + noise
            if s.get("is_integer", False):
                v = int(round(v))
                v = max(int(s.get("min", v)), min(int(s.get("max", v)), v))
            else:
                v = round(max(s.get("min", v), min(s.get("max", v), v)), 4)
            row[col] = v
        row[target_col] = target_class
        synthetic.append(row)

    return synthetic


PYTHON_TECHNIQUES = {"smote": python_smote, "gaussian_noise": python_gaussian_noise}


# Per-model augmentation driver

def augment_with_model(
    friendly_name, model_tag, batch_size, use_completion_prompt, fallback_technique,
    max_empty_batches, stream_timeout,
    columns, stats_minority, minority_rows, minority_samples,
    target_col, target_class, rows_needed, ollama_url, out_dir, log_dir,
) -> tuple:
    """
    Run the two-step technique-selection + batched-generation loop for one
    model, falling back to a Python generator if the model cannot produce
    valid rows. Returns (synthetic_rows, technique_explanation_text).
    """
    out_dir = Path(out_dir)
    log_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model %s (%s): choosing augmentation technique...", friendly_name, model_tag)
    try:
        tech_raw = call_with_retry(
            ollama_url, model_tag, build_technique_prompt(columns, stats_minority), max_seconds=stream_timeout
        )
    except Exception as e:
        logger.error("Step 1 failed for %s: %s", friendly_name, e)
        return [], ""

    tech_clean = strip_think_tags(tech_raw).strip()
    (log_dir / "technique.txt").write_text(tech_clean, encoding="utf-8")
    technique_name = tech_clean.split(".")[0][:80].strip()

    collected = []
    batch_num = 0
    consecutive_empty = 0

    logger.info("Generating %d rows for %s (batches of %d)...", rows_needed, friendly_name, batch_size)
    while len(collected) < rows_needed:
        remaining = rows_needed - len(collected)
        this_batch = min(batch_size, remaining)
        batch_num += 1

        data_prompt = (
            build_completion_prompt(columns, minority_samples)
            if use_completion_prompt
            else build_data_prompt(columns, stats_minority, minority_samples, technique_name, this_batch)
        )

        try:
            data_raw = call_with_retry(ollama_url, model_tag, data_prompt, max_seconds=stream_timeout)
        except Exception as e:
            logger.warning("Batch %d failed for %s: %s", batch_num, friendly_name, e)
            time.sleep(10)
            consecutive_empty += 1
            if consecutive_empty >= 5:
                logger.warning("5 consecutive failures, skipping %s.", friendly_name)
                break
            continue

        (log_dir / f"batch_{batch_num:03d}.txt").write_text(data_raw, encoding="utf-8")

        raw_rows = extract_rows(data_raw, n_cols=len(columns))
        valid_rows = validate_rows(raw_rows, columns, stats_minority, target_col, target_class)
        collected.extend(valid_rows)
        logger.info("Batch %d: got %d valid rows (total %d)", batch_num, len(valid_rows), len(collected))

        if len(valid_rows) == 0:
            consecutive_empty += 1
            if consecutive_empty >= max_empty_batches:
                logger.info("%d consecutive empty batches - triggering Python fallback.", max_empty_batches)
                break
            time.sleep(3)
        else:
            consecutive_empty = 0

    collected = collected[:rows_needed]

    if len(collected) == 0:
        fn = PYTHON_TECHNIQUES.get(fallback_technique, python_smote)
        logger.info("LLM produced no valid rows for %s; running Python %s fallback.", friendly_name, fallback_technique)
        collected = fn(
            minority_rows=minority_rows, columns=columns, stats=stats_minority,
            target_col=target_col, target_class=target_class, n_rows=rows_needed,
        )
        (log_dir / "fallback_note.txt").write_text(
            f"LLM ({friendly_name} / {model_tag}) could not produce valid CSV rows.\n"
            f"Technique chosen by model (Step 1): {tech_clean[:500]}\n\n"
            f"Python fallback used: {fallback_technique}\n"
            f"Generated {len(collected)} rows using Python implementation.",
            encoding="utf-8",
        )

    syn_path = out_dir / "synthetic_rows.csv"
    with open(syn_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(collected)
    logger.info("Saved %s synthetic rows to %s", friendly_name, syn_path)

    return collected, tech_clean


def is_ollama_reachable(ollama_url: str = DEFAULT_OLLAMA_URL, timeout: float = 3.0) -> bool:
    """Quick reachability probe used by main.py to decide whether to attempt this stage."""
    try:
        requests.get(f"{ollama_url}/api/tags", timeout=timeout)
        return True
    except requests.RequestException:
        return False


def run_llm_augmentation(
    input_csv: Path,
    out_root: Path,
    log_root: Path,
    target_col: str = DEFAULT_TARGET_COL,
    minority_class: int = DEFAULT_MINORITY_CLASS,
    rows_per_model: int = 300,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    models: tuple = ("mistral", "phi"),
) -> dict:
    """
    Run mistral + phi augmentation over `input_csv` and write, per model,
    outputs/<model>/synthetic_rows.csv and outputs/<model>/combined_dataset.csv.

    Parameters
    ----------
    input_csv : Path
        Training CSV with a binary `target_col` column.
    out_root, log_root : Path
        Root directories for per-model outputs and raw-response logs.
    rows_per_model : int
        Synthetic rows to generate per model.
    ollama_url : str
        Base URL of the Ollama server.
    models : tuple
        Subset of ("mistral", "phi") to run.

    Returns
    -------
    dict
        {model_name: {"n_rows": int, "combined_path": Path}}
    """
    out_root = Path(out_root)
    log_root = Path(log_root)
    columns, all_rows = load_csv(input_csv)
    minority_rows = [r for r in all_rows if r.get(target_col) == str(minority_class)]
    stats_min = compute_stats(minority_rows, columns)
    samples = sample_rows(minority_rows, n=6)

    logger.info(
        "LLM augmentation: %d rows, %d columns, %d minority rows",
        len(all_rows), len(columns), len(minority_rows),
    )

    results = {}
    selected = [(name, MODELS[name]) for name in models if name in MODELS]
    for i, (name, (tag, batch_sz, use_completion, fallback_tech, max_empty, stream_tout)) in enumerate(selected):
        out_dir = out_root / name
        log_dir = log_root / name

        synthetic, technique = augment_with_model(
            friendly_name=name, model_tag=tag, batch_size=batch_sz,
            use_completion_prompt=use_completion, fallback_technique=fallback_tech,
            max_empty_batches=max_empty, stream_timeout=stream_tout,
            columns=columns, stats_minority=stats_min, minority_rows=minority_rows,
            minority_samples=samples, target_col=target_col, target_class=minority_class,
            rows_needed=rows_per_model, ollama_url=ollama_url, out_dir=out_dir, log_dir=log_dir,
        )

        combined = [{c: r[c] for c in columns} for r in all_rows]
        combined += [{c: r.get(c, "") for c in columns} for r in synthetic]
        combined_path = out_dir / "combined_dataset.csv"
        with open(combined_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(combined)

        results[name] = {"n_rows": len(synthetic), "combined_path": combined_path, "technique": technique}

        if i < len(selected) - 1:
            time.sleep(INTER_MODEL_DELAY)

    return results
