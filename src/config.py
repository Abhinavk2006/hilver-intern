import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLDEN_SET_DIR = DATA_DIR / "golden_set"
RESULTS_DIR = BASE_DIR / "results"

TWCS_CSV_PATH = RAW_DATA_DIR / "twcs.csv"
PROCESSED_BRAND_DATA_PATH = PROCESSED_DATA_DIR / "brand_conversations.json"
GOLDEN_SET_PATH = GOLDEN_SET_DIR / "golden_set.json"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42
