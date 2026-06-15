"""
Embedding-based similarity between article text and community summaries.

Loads pre-built fake / real community summaries from JSON or parquet,
embeds them via an OpenAI-compatible embedding API (default port 8002), and
exposes cosine-similarity search at prediction time.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from narrative_map.config import COMMUNITIES_DIR, DEFAULT_EMBED_URL


def _clean_llm_artifacts(text: str) -> str:
    """Strip leaked thinking / completion tokens from LLM-generated text."""
    if not text:
        return text
    text = re.split(r"#{5,}", text)[0]
    text = re.split(r"\[\[.*?completed.*?\]\]", text)[0]
    text = re.sub(r"<｜[^>]*>?", "", text)
    return text.rstrip()


def load_communities(path: Path) -> pd.DataFrame:
    """Load community reports from a JSON or parquet file.

    JSON files are expected to be a list of dicts with at least
    ``title`` and ``summary`` fields.  Parquet files are read as-is.
    Both formats are normalised to a DataFrame with a ``findings``
    column (list of dicts).
    """
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for entry in data:
            rows.append({
                "community_id": entry.get("community_id", 0),
                "level": entry.get("level", 0),
                "title": _clean_llm_artifacts(str(entry.get("title", ""))),
                "summary": _clean_llm_artifacts(str(entry.get("summary", ""))),
                "findings": [],
                "rating": entry.get("rating", 0.0),
                "rating_explanation": _clean_llm_artifacts(
                    str(entry.get("rating_explanation", ""))
                ),
                "full_content": _clean_llm_artifacts(
                    str(entry.get("full_content", ""))
                ),
            })
        return pd.DataFrame(rows)

    return pd.read_parquet(path)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _resolve_community_file(
    communities_dir: Path, label: str,
) -> Path:
    """Find the community file for *label* (``fake`` or ``real``).

    Checks, in order: ``<label>.json``, ``<label>.parquet``.
    """
    for ext in (".json", ".parquet"):
        p = communities_dir / f"{label}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No community file found for '{label}' in {communities_dir}"
    )


class CommunitySimilarity:
    """Embed community summaries once; query by article text at prediction time."""

    BATCH_SIZE = 32

    def __init__(
        self,
        communities_dir: Path = COMMUNITIES_DIR,
        embed_base_url: str = DEFAULT_EMBED_URL,
        embed_model: str | None = None,
        fake_path: Path | None = None,
        real_path: Path | None = None,
    ):
        self.client = OpenAI(base_url=embed_base_url, api_key="no-key")

        if embed_model is None:
            models = self.client.models.list()
            self.embed_model = models.data[0].id if models.data else "default"
        else:
            self.embed_model = embed_model

        fake_file = fake_path or _resolve_community_file(communities_dir, "fake")
        real_file = real_path or _resolve_community_file(communities_dir, "real")
        print(f"  Loading communities: {fake_file.name}, {real_file.name}")

        self.fake_df = load_communities(fake_file)
        self.real_df = load_communities(real_file)

        fake_texts = self.fake_df["summary"].fillna("").tolist()
        real_texts = self.real_df["summary"].fillna("").tolist()

        print(
            f"  Encoding {len(fake_texts)} fake + "
            f"{len(real_texts)} real community summaries "
            f"(model={self.embed_model}) …"
        )
        self.fake_emb = self._embed_batch(fake_texts)
        self.real_emb = self._embed_batch(real_texts)
        print(
            f"  ✓ Community embeddings ready (dim={self.fake_emb.shape[1]})"
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts in batches via the embedding API."""
        all_embs: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            resp = self.client.embeddings.create(
                model=self.embed_model,
                input=batch,
            )
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            all_embs.extend(d.embedding for d in sorted_data)
        arr = np.array(all_embs, dtype=np.float32)
        return _normalize(arr)

    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=[text],
        )
        arr = np.array([resp.data[0].embedding], dtype=np.float32)
        return _normalize(arr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(df: pd.DataFrame, idx: int, sims: np.ndarray) -> dict:
        row = df.iloc[int(idx)]
        findings_raw = row.get("findings")
        findings: list[str] = []
        if findings_raw is not None:
            try:
                for f in findings_raw:
                    if isinstance(f, dict):
                        findings.append(f.get("summary", ""))
            except (TypeError, KeyError):
                pass
        return {
            "title": row["title"],
            "summary": str(row.get("summary", "")),
            "findings": findings[:5],
            "similarity": float(sims[int(idx)]),
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = 3) -> dict:
        """Return top-k similar communities from each set + statistics.

        Parameters
        ----------
        text : str
            Article text (headline + body) to compare against communities.
        top_k : int
            Number of most-similar communities to return per set.

        Returns
        -------
        dict with keys ``top_fake``, ``top_real``, ``stats``.
        """
        emb = self._embed_single(text)

        fake_sims = (self.fake_emb @ emb.T).flatten()
        real_sims = (self.real_emb @ emb.T).flatten()

        fake_idx = np.argsort(fake_sims)[::-1][:top_k]
        real_idx = np.argsort(real_sims)[::-1][:top_k]

        top_fake = [self._row_to_dict(self.fake_df, i, fake_sims) for i in fake_idx]
        top_real = [self._row_to_dict(self.real_df, i, real_sims) for i in real_idx]

        stats = {
            "fake_max_sim": float(fake_sims.max()),
            "fake_mean_sim": float(fake_sims.mean()),
            "fake_std_sim": float(fake_sims.std()),
            "real_max_sim": float(real_sims.max()),
            "real_mean_sim": float(real_sims.mean()),
            "real_std_sim": float(real_sims.std()),
            "diff_max_sim": float(real_sims.max() - fake_sims.max()),
            "diff_mean_sim": float(real_sims.mean() - fake_sims.mean()),
        }

        return {"top_fake": top_fake, "top_real": top_real, "stats": stats}
