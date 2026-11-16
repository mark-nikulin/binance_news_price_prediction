"""Baseline ML models and inference."""

from news_pred.models.baseline import CatBoostTrendModel, TrainMetrics
from news_pred.models.inference import generate_signals, save_signals

__all__ = ["CatBoostTrendModel", "TrainMetrics", "generate_signals", "save_signals"]

