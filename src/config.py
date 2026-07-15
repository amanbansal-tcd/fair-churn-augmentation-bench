"""
config.py
=========
Single source of truth for every path, constant, and hyperparameter used
across the pipeline.  Import from here; never hard-code values elsewhere.

Layout
------
  Paths & directories
  Dataset schema constants
  Encoding maps
  Model hyperparameters
  Augmentation settings
  Fairness thresholds
  Plot styling
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED: int = 42

# ══════════════════════════════════════════════════════════════════════════════
# Directory layout
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR      = Path(__file__).resolve().parent.parent   # project root
DATA_DIR      = ROOT_DIR / "data"
SRC_DIR       = ROOT_DIR / "src"
OUT_DIR       = ROOT_DIR / "outputs"
FIG_DIR       = OUT_DIR  / "figures"                     # all PNG figures
MODEL_DIR     = OUT_DIR  / "models"                      # joblib .pkl files
REPORT_DIR    = OUT_DIR  / "reports"                     # CSV metric tables
AUG_DIR       = OUT_DIR  / "augmented_datasets"          # augmented CSV exports
RAW_DATA_PATH = DATA_DIR / "NewspaperChurn.csv"

# ══════════════════════════════════════════════════════════════════════════════
# Dataset schema
# ══════════════════════════════════════════════════════════════════════════════

TARGET_COL  = "Subscriber"     # raw column name in CSV
TARGET_POS  = "YES"            # positive class label
ID_COL      = "SubscriptionID"

# Columns dropped before modelling (identifiers and geography)
DROP_COLS = ["SubscriptionID", "Address", "State", "City", "County", "Zip Code"]

# Sensitive attributes examined during the fairness audit
SENSITIVE_ATTRS = ["Ethnicity", "Language", "Home Ownership", "Age range"]

# Encoded feature names that represent each sensitive group
# (used by the bias-mitigation feature-removal experiments)
SENSITIVE_FEATS = {
    "ETHNICITY": ["Ethnicity_enc"],
    "LANGUAGE":  ["Language_enc"],
    "HOMEOWN":   ["Home Ownership_enc"],
    "GEOGRAPHY": [],            # geography already removed via DROP_COLS
}

# ══════════════════════════════════════════════════════════════════════════════
# Engineered / encoded feature list (model inputs after preprocessing)
# ══════════════════════════════════════════════════════════════════════════════

ALL_FEATURES: list[str] = [
    "income_ord",               # ordinal HH income (0–15)
    "age_mid",                  # age-band midpoint
    "fee_mid",                  # weekly fee midpoint ($)
    "Year Of Residence",        # numeric (raw)
    "reward_prog_w",            # reward pts, winsorised at 99th pct
    "Home Ownership_enc",       # label-encoded
    "Ethnicity_enc",            # label-encoded
    "Language_enc",             # label-encoded
    "dummy for Children_enc",   # label-encoded (Y/N)
    "DP_enc",                   # delivery period label-encoded
    "Nielsen Prizm_enc",        # label-encoded
    "Source Channel_enc",       # label-encoded
]

# Human-readable labels used in figures
FEATURE_LABELS: dict[str, str] = {
    "income_ord":             "HH Income (Ordinal)",
    "age_mid":                "Age (Midpoint)",
    "fee_mid":                "Weekly Fee ($)",
    "Year Of Residence":      "Years of Residence",
    "reward_prog_w":          "Reward Programme",
    "Home Ownership_enc":     "Home Ownership",
    "Ethnicity_enc":          "Ethnicity",
    "Language_enc":           "Language",
    "dummy for Children_enc": "Has Children",
    "DP_enc":                 "Delivery Period",
    "Nielsen Prizm_enc":      "Nielsen PRIZM",
    "Source Channel_enc":     "Source Channel",
}

# ══════════════════════════════════════════════════════════════════════════════
# Train / test split
# ══════════════════════════════════════════════════════════════════════════════

TEST_SIZE = 0.20   # 80/20 stratified split
CV_FOLDS  = 5      # stratified k-fold for cross-validation

# ══════════════════════════════════════════════════════════════════════════════
# Encoding lookup maps (used in preprocessing.py)
# ══════════════════════════════════════════════════════════════════════════════

INCOME_ORDER: list[str] = [
    "Under $20,000",       "$  20,000 - $29,999", "$  30,000 - $39,999",
    "$  40,000 - $49,999", "$  50,000 - $59,999", "$  60,000 - $69,999",
    "$  70,000 - $79,999", "$  80,000 - $89,999", "$  90,000 - $99,999",
    "$100,000 - $124,999", "$125,000 - $149,999", "$150,000 - $174,999",
    "$175,000 - $199,999", "$200,000 - $249,999", "$250,000 - $499,999",
    "$500,000 Plus",
]
INCOME_MAP: dict[str, int] = {v: i for i, v in enumerate(INCOME_ORDER)}

AGE_MIDPOINTS: dict[str, float] = {
    "18-21": 19.0, "22-24": 23.0, "25-29": 27.0, "30-34": 32.0, "35-39": 37.0,
    "40-44": 42.0, "45-49": 47.0, "50-54": 52.0, "55-59": 57.0, "60-64": 62.0,
    "65-69": 67.0, "70-74": 72.0, "75 years or more": 78.0, "Unknown": 52.0,
}

# Delivery-period label normalisation (consolidates legacy codes)
DP_NORMALISE: dict[str, str] = {
    "SoooTFS":  "Thu-Sun", "Soooooo":  "7Day",    "7DayOL":  "7Day",
    "7DayT":    "7Day",    "THU-SUN":  "Thu-Sun",  "SatSun":  "Sat-Sun",
    "Sooooo":   "7Day",    "SooooTFS": "Thu-Sun",
}

# ══════════════════════════════════════════════════════════════════════════════
# Model hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

LR_PARAMS: dict = dict(max_iter=1000, random_state=RANDOM_SEED, C=0.5)

RF_PARAMS: dict = dict(
    n_estimators=400, max_depth=12, min_samples_leaf=10,
    random_state=RANDOM_SEED, n_jobs=-1,
)

# XGBoost 3.x: eval_metric is accepted as a **kwarg
XGB_PARAMS: dict = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0,
)

LGB_PARAMS: dict = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbose=-1,
)

# M12 final model — slightly tuned for the debiased feature set
LGB_FINAL_PARAMS: dict = dict(
    n_estimators=500, max_depth=6, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    random_state=RANDOM_SEED, verbose=-1,
)

CAT_PARAMS: dict = dict(
    iterations=400, depth=6, learning_rate=0.05,
    random_seed=RANDOM_SEED, verbose=False,
)

# ══════════════════════════════════════════════════════════════════════════════
# Augmentation settings
# ══════════════════════════════════════════════════════════════════════════════

# Target minority-class size for SMOTE / ADASYN / Gaussian oversampling.
# The actual count may be slightly different for ADASYN (adaptive).
AUG_TARGET_MINORITY: int   = 6_000

AUG_SMOTE_K:          int   = 5      # SMOTE k_neighbors
AUG_ADASYN_K:         int   = 5      # ADASYN n_neighbors
AUG_GAUSS_NOISE_SCALE: float = 0.05  # fraction of per-feature std for Gaussian
AUG_GAUSS_MAJ_FRAC:   float = 0.30  # fraction of majority to duplicate with noise

# CTGAN settings (reduce EPOCHS to ~50 for quick test runs)
AUG_CTGAN_EPOCHS:     int = 150
AUG_CTGAN_BATCH_SIZE: int = 500
AUG_CTGAN_N_SYNTH:    int = 3_570   # synthetic minority rows to generate

# Per-feature valid range used for post-processing clipping / rounding
FEATURE_RANGES: dict[str, dict] = {
    "income_ord":             {"type": "int",   "min": 0,    "max": 15},
    "age_mid":                {"type": "float", "min": 18.0, "max": 85.0},
    "fee_mid":                {"type": "float", "min": 0.0,  "max": 20.0},
    "Year Of Residence":      {"type": "int",   "min": 0,    "max": 50},
    "reward_prog_w":          {"type": "float", "min": 0.0,  "max": 5000.0},
    "Home Ownership_enc":     {"type": "int",   "min": 0,    "max": 4},
    "Ethnicity_enc":          {"type": "int",   "min": 0,    "max": 72},
    "Language_enc":           {"type": "int",   "min": 0,    "max": 30},
    "dummy for Children_enc": {"type": "int",   "min": 0,    "max": 2},
    "DP_enc":                 {"type": "int",   "min": 0,    "max": 8},
    "Nielsen Prizm_enc":      {"type": "int",   "min": 0,    "max": 9},
    "Source Channel_enc":     {"type": "int",   "min": 0,    "max": 20},
}

# ══════════════════════════════════════════════════════════════════════════════
# Fairness thresholds
# ══════════════════════════════════════════════════════════════════════════════

FAIRNESS_THRESHOLDS: dict = {
    "DPD_severe":  0.20,   # Demographic Parity Diff flagged as severe
    "DIR_legal":   0.80,   # EEOC 4/5ths (80%) rule for Disparate Impact Ratio
    "MIN_GROUP_N": 30,     # minimum group size for reliable fairness metrics
}

# ══════════════════════════════════════════════════════════════════════════════
# Plot styling
# ══════════════════════════════════════════════════════════════════════════════

PALETTE: dict[str, str] = {
    "YES":    "#27ae60",
    "NO":     "#c0392b",
    "blue":   "#2980b9",
    "orange": "#e67e22",
    "purple": "#8e44ad",
    "green":  "#27ae60",
    "red":    "#e74c3c",
}

MODEL_COLORS: dict[str, str] = {
    "Logistic Regression": "#3498db",
    "Random Forest":       "#27ae60",
    "XGBoost":             "#e67e22",
    "LightGBM":            "#8e44ad",
    "CatBoost":            "#e74c3c",
}

# Colour palette for augmentation-comparison figures
AUG_COLORS: dict[str, str] = {
    "original":       "#2c3e50",
    "SMOTE":          "#2980b9",
    "ADASYN":         "#27ae60",
    "Gaussian Noise": "#e67e22",
    "CTGAN":          "#8e44ad",
}

FIG_DPI: int = 150   # resolution for all saved figures
