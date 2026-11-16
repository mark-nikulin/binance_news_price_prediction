"""Market features, Event Study, and ML dataset construction."""

from news_pred.features.dataset import build_training_dataset
from news_pred.features.event_study import run_event_study, summarize_event_study
from news_pred.features.market import compute_market_features, load_bars

__all__ = [
    "load_bars",
    "compute_market_features",
    "run_event_study",
    "summarize_event_study",
    "build_training_dataset",
]

