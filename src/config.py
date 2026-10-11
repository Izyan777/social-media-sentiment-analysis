"""Central place for paths and constants so every script agrees on them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
RAW_DIR = DATA_DIR / "raw"

REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# Public Roman Urdu sentiment dataset (GPL-3.0). We download it, we do not re-host it.
ROMAN_URDU_REPO_URL = "https://github.com/Smat26/Roman-Urdu-Dataset.git"
ROMAN_URDU_REPO_DIR = EXTERNAL_DIR / "Roman-Urdu-Dataset"
ROMAN_URDU_CSV = ROMAN_URDU_REPO_DIR / "Dataset" / "Roman Urdu DataSet.csv"

CLEAN_CSV = PROCESSED_DIR / "roman_urdu_clean.csv"
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"

LABELS = ["Negative", "Neutral", "Positive"]
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for d in (EXTERNAL_DIR, PROCESSED_DIR, SAMPLE_DIR, RAW_DIR, FIG_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
