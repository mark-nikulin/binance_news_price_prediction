"""Minimal CatBoost baseline with chronological split and purge gap."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from news_pred.features.dataset import MARKET_FEATURE_COLS
from news_pred.features.text import TextPCATransformer
from news_pred.paths import MODELS_DIR

FeatureMode = Literal["market_only", "market_plus_news", "market_plus_news_text"]


@dataclass
class TrainMetrics:
    feature_mode: str
    train_rows: int
    test_rows: int
    test_mae: float
    test_r2: float
    test_dir_accuracy: float
    split_ts: str


def chronological_purged_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    purge_minutes: int = 240,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split dataset chronologically with a purge gap >= horizon to prevent target leakage."""
    ordered = df.sort_values("published_ts").reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * train_frac))
    split_ts = pd.Timestamp(ordered.loc[split_idx, "published_ts"])
    purge_delta = pd.Timedelta(minutes=purge_minutes)

    train_df = ordered[ordered["published_ts"] < split_ts - purge_delta].copy()
    test_df = ordered[ordered["published_ts"] >= split_ts].copy()
    return train_df, test_df, split_ts


class CatBoostTrendModel:
    """Wrapper around CatBoostRegressor for predicting forward returns from market + news features."""

    def __init__(
        self,
        feature_mode: FeatureMode = "market_plus_news",
        horizon_minutes: int = 240,
        iterations: int = 150,
        depth: int = 3,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 10.0,
        loss_function: str = "Huber:delta=0.05",
        random_seed: int = 42,
    ):
        self.feature_mode = feature_mode
        self.horizon_minutes = horizon_minutes
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.random_seed = random_seed

        self.feature_cols: List[str] = list(MARKET_FEATURE_COLS)
        self.cat_features: List[str] = []
        self.text_pca: Optional[TextPCATransformer] = None
        if feature_mode in ("market_plus_news", "market_plus_news_text"):
            self.feature_cols.append("event_type")
            self.cat_features.append("event_type")
        if feature_mode == "market_plus_news_text":
            self.text_pca = TextPCATransformer(n_components=8)

        self.model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            loss_function=loss_function,
            verbose=False,
        )

    def _prepare_X(self, df: pd.DataFrame, fit_text: bool = False) -> pd.DataFrame:
        X = df[MARKET_FEATURE_COLS + self.cat_features].copy()
        for col in MARKET_FEATURE_COLS:
            X[col] = X[col].astype(float).fillna(0.0)
        for col in self.cat_features:
            X[col] = X[col].astype(str).fillna("other")
        if self.text_pca is not None:
            text_df = (
                self.text_pca.fit_transform(df["title"])
                if fit_text
                else self.text_pca.transform(df["title"])
            )
            text_df.index = X.index
            for col in text_df.columns:
                X[col] = text_df[col].astype(float)
            if fit_text:
                self.feature_cols = list(X.columns)
        return X

    def fit_and_evaluate(
        self,
        dataset_df: pd.DataFrame,
        train_frac: float = 0.7,
    ) -> Tuple[TrainMetrics, pd.DataFrame]:
        valid = dataset_df.dropna(subset=["fwd_ret"]).copy()
        if len(valid) < 10:
            raise ValueError(f"Not enough valid rows to train ({len(valid)} < 10)")

        train_df, test_df, split_ts = chronological_purged_split(
            valid, train_frac=train_frac, purge_minutes=self.horizon_minutes
        )
        if train_df.empty or test_df.empty:
            raise ValueError("Train or test split is empty after purge gap")

        X_train = self._prepare_X(train_df, fit_text=True)
        y_train = train_df["fwd_ret"].astype(float).to_numpy()

        X_test = self._prepare_X(test_df, fit_text=False)
        y_test = test_df["fwd_ret"].astype(float).to_numpy()

        self.model.fit(X_train, y_train, cat_features=self.cat_features)
        preds = self.model.predict(X_test)

        test_out = test_df.copy()
        test_out["pred_ret"] = preds

        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))
        metrics = TrainMetrics(
            feature_mode=self.feature_mode,
            train_rows=len(train_df),
            test_rows=len(test_df),
            test_mae=float(mean_absolute_error(y_test, preds)),
            test_r2=float(r2_score(y_test, preds)) if len(y_test) > 1 else 0.0,
            test_dir_accuracy=dir_acc,
            split_ts=split_ts.isoformat(),
        )
        return metrics, test_out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_X(df)
        return np.asarray(self.model.predict(X), dtype=float)

    def save(self, name: str = "catboost_baseline", models_dir: Path = MODELS_DIR) -> Path:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{name}.cbm"
        meta_path = models_dir / f"{name}.meta.json"

        self.model.save_model(str(model_path))
        meta: Dict[str, object] = {
            "feature_mode": self.feature_mode,
            "horizon_minutes": self.horizon_minutes,
            "feature_cols": self.feature_cols,
            "cat_features": self.cat_features,
            "text_pca": self.text_pca.to_dict() if self.text_pca is not None else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return model_path

    @classmethod
    def load(cls, name: str = "catboost_baseline", models_dir: Path = MODELS_DIR) -> "CatBoostTrendModel":
        models_dir = Path(models_dir)
        model_path = models_dir / f"{name}.cbm"
        meta_path = models_dir / f"{name}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        obj = cls(
            feature_mode=meta["feature_mode"],
            horizon_minutes=int(meta["horizon_minutes"]),
        )
        obj.feature_cols = list(meta["feature_cols"])
        if meta.get("text_pca"):
            obj.text_pca = TextPCATransformer.from_dict(meta["text_pca"])
        obj.model.load_model(str(model_path))
        return obj

