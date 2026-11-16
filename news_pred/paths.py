"""Default filesystem paths for news, market data, features, models, and signals."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("NP_DATA_DIR", ROOT_DIR / "data"))

RAW_NEWS_DIR = DATA_DIR / "raw" / "news"
NORMALIZED_NEWS_DIR = DATA_DIR / "normalized" / "news"
MARKET_DIR = DATA_DIR / "market"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
SIGNALS_DIR = DATA_DIR / "signals"
REPORTS_DIR = DATA_DIR / "reports"

