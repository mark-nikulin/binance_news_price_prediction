"""Inference: convert model predictions into SIGNAL_SCHEMA-compliant trading signals."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from news_pred.backtest.contract import SIGNAL_SCHEMA
from news_pred.models.baseline import CatBoostTrendModel
from news_pred.paths import SIGNALS_DIR


def generate_signals(
    model: CatBoostTrendModel,
    features_df: pd.DataFrame,
    score_threshold: float = 0.002,
    entry_delay_minutes: int = 5,
) -> pd.DataFrame:
    """
    Generate trading signals conforming to SIGNAL_SCHEMA from features DataFrame.
    `score_threshold` (default 0.002 = 20 bps) filters weak predictions to direction=0.
    """
    if features_df.empty:
        return pd.DataFrame(columns=SIGNAL_SCHEMA.names)

    scores = model.predict(features_df)
    directions = np.where(
        scores > score_threshold,
        1,
        np.where(scores < -score_threshold, -1, 0),
    ).astype("int8")

    signal_ts = (
        pd.to_datetime(features_df["published_ts"], utc=True)
        + pd.Timedelta(minutes=entry_delay_minutes)
    )
    horizon_ns = int(model.horizon_minutes) * 60 * 1_000_000_000

    out = pd.DataFrame(
        {
            "signal_ts": signal_ts,
            "symbol": features_df["okx_symbol"].astype(str),
            "direction": directions,
            "score": scores.astype("float64"),
            "horizon_ns": np.full(len(features_df), horizon_ns, dtype="int64"),
            "news_id": features_df["news_id"].astype(str),
            "event_type": features_df["event_type"].astype(str),
        }
    )
    return out.sort_values("signal_ts").reset_index(drop=True)


def save_signals(signals_df: pd.DataFrame, filename: str = "signals.parquet", signals_dir: Path = SIGNALS_DIR) -> Path:
    signals_dir = Path(signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)
    target = signals_dir / filename
    table = pa.Table.from_pandas(signals_df, schema=SIGNAL_SCHEMA, preserve_index=False)
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(target)
    return target

