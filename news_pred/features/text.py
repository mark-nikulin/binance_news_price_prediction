"""BERT/MiniLM text embeddings with leak-free PCA compression."""

import hashlib
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from news_pred.paths import FEATURES_DIR

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _title_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def encode_titles(
    titles: Iterable[str],
    model_name: str = DEFAULT_EMBED_MODEL,
    cache_dir: Path = FEATURES_DIR,
) -> np.ndarray:
    """
    Encode news titles into dense embeddings using a frozen pretrained transformer.
    Results are cached by title hash so repeated runs do not re-run inference.
    """
    title_list = [str(t) for t in titles]
    if not title_list:
        return np.empty((0, 0), dtype=np.float32)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "__")
    cache_file = cache_dir / f"emb_{safe_name}.parquet"

    cached_map = {}
    if cache_file.exists():
        cdf = pd.read_parquet(cache_file)
        for h, vec in zip(cdf["hash"], cdf["emb"]):
            cached_map[h] = np.asarray(vec, dtype=np.float32)

    hashes = [_title_hash(t) for t in title_list]
    missing_idx = [i for i, h in enumerate(hashes) if h not in cached_map]

    if missing_idx:
        from sentence_transformers import SentenceTransformer  # lazy import

        logger.info("Encoding %d new titles with %s", len(missing_idx), model_name)
        st_model = SentenceTransformer(model_name)
        missing_texts = [title_list[i] for i in missing_idx]
        new_vecs = st_model.encode(missing_texts, show_progress_bar=False, normalize_embeddings=True)
        for idx, vec in zip(missing_idx, new_vecs):
            cached_map[hashes[idx]] = np.asarray(vec, dtype=np.float32)

        out_df = pd.DataFrame(
            {
                "hash": list(cached_map.keys()),
                "emb": [v.tolist() for v in cached_map.values()],
            }
        )
        tmp = cache_file.with_suffix(".parquet.tmp")
        out_df.to_parquet(tmp, index=False)
        tmp.replace(cache_file)

    return np.vstack([cached_map[h] for h in hashes]).astype(np.float32)


class TextPCATransformer:
    """Fits PCA strictly on training embeddings and projects to `text_pca_0..k`."""

    def __init__(self, n_components: int = 8, model_name: str = DEFAULT_EMBED_MODEL):
        self.n_components = n_components
        self.model_name = model_name
        self._pca: Optional[PCA] = None
        self.cols: List[str] = []

    def fit_transform(self, titles: Iterable[str]) -> pd.DataFrame:
        emb = encode_titles(titles, model_name=self.model_name)
        k = min(self.n_components, emb.shape[0] - 1, emb.shape[1])
        k = max(1, k)
        self._pca = PCA(n_components=k, random_state=42)
        proj = self._pca.fit_transform(emb)
        self.cols = [f"text_pca_{i}" for i in range(k)]
        return pd.DataFrame(proj, columns=self.cols)

    def transform(self, titles: Iterable[str]) -> pd.DataFrame:
        if self._pca is None:
            raise RuntimeError("TextPCATransformer must be fitted before transform()")
        emb = encode_titles(titles, model_name=self.model_name)
        proj = self._pca.transform(emb)
        return pd.DataFrame(proj, columns=self.cols)

    def to_dict(self) -> dict:
        if self._pca is None:
            return {}
        return {
            "n_components": self.n_components,
            "model_name": self.model_name,
            "cols": self.cols,
            "mean": self._pca.mean_.tolist(),
            "components": self._pca.components_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextPCATransformer":
        obj = cls(n_components=int(data["n_components"]), model_name=str(data["model_name"]))
        pca = PCA(n_components=len(data["cols"]), random_state=42)
        pca.mean_ = np.asarray(data["mean"], dtype=np.float64)
        pca.components_ = np.asarray(data["components"], dtype=np.float64)
        pca.n_components_ = len(data["cols"])
        obj._pca = pca
        obj.cols = list(data["cols"])
        return obj

