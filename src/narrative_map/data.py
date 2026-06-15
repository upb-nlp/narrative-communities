"""Unified dataset loading for indexing, analysis, and evaluation."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from datasets import load_dataset

from narrative_map.config import DATASET_NAME, REAL_NEWS_LABEL

logger = logging.getLogger(__name__)

DatasetSplit = Literal["train", "validation", "test"]

HF_BASE_URL = f"hf://datasets/{DATASET_NAME}/"
HF_SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def create_text_from_row(row: dict) -> str | None:
    """Format headline + body from a dataset row."""
    parts: list[str] = []
    headline = row.get("headline")
    if headline and str(headline).strip() not in ("null", ""):
        parts.append(str(headline).strip())
    body = row.get("body")
    if body and str(body).strip() not in ("null", ""):
        parts.append(str(body).strip())
    return "\n\n".join(parts) if parts else None


def load_dataset_split(
    split: DatasetSplit = "test",
    *,
    limit: int | None = None,
    balanced: bool = False,
) -> list[dict]:
    """Load one official HF split (train, validation, or test)."""
    import random

    print(f"Loading {split} data from {DATASET_NAME}…")
    ds = load_dataset(DATASET_NAME, "default")

    if split in ds:
        samples = ds[split].to_list()
    elif split == "test" and "train" in ds:
        print("  No test split; using last 20 % of train data")
        train_data = ds["train"].to_list()
        split_idx = int(len(train_data) * 0.8)
        samples = train_data[split_idx:]
    else:
        raise ValueError(
            f"Split {split!r} not found in dataset (available: {list(ds.keys())})"
        )

    real_news = [s for s in samples if s.get("supernarrative") == REAL_NEWS_LABEL]
    fake_news = [s for s in samples if s.get("supernarrative") != REAL_NEWS_LABEL]
    print(f"  Total: {len(samples)}  (real={len(real_news)}, fake={len(fake_news)})")

    if balanced:
        min_count = min(len(real_news), len(fake_news))
        if limit:
            min_count = min(min_count, limit // 2)
        random.seed(42)
        random.shuffle(real_news)
        random.shuffle(fake_news)
        balanced_data = real_news[:min_count] + fake_news[:min_count]
        random.shuffle(balanced_data)
        print(f"  Balanced to {min_count} + {min_count} = {len(balanced_data)}")
        return balanced_data

    if limit:
        samples = samples[:limit]
        print(f"  Limited to {limit}")
    return samples


def load_split_ids(split: DatasetSplit) -> set[str]:
    """Return stable article ids for an official split."""
    samples = load_dataset_split(split)
    ids: set[str] = set()
    for sample in samples:
        sid = sample.get("id")
        if sid is not None:
            ids.add(str(sid))
    return ids


def ground_truth_label(sample: dict) -> str:
    return "real" if sample.get("supernarrative", "") == REAL_NEWS_LABEL else "fake"


def load_indexing_corpus(*, fake_only: bool = True) -> tuple[list[str], pd.DataFrame]:
    """Load train+validation documents for GraphRAG indexing.

    By default keeps fake news only (excludes ``REAL_NEWS_LABEL``).
    """
    logger.info("Loading train and validation splits for indexing…")
    train_df = pd.read_parquet(HF_BASE_URL + HF_SPLITS["train"])
    validation_df = pd.read_parquet(HF_BASE_URL + HF_SPLITS["validation"])
    labeled_df = pd.concat([train_df, validation_df], ignore_index=True)

    if fake_only:
        labeled_df = labeled_df[
            labeled_df["supernarrative"].str.lower() != REAL_NEWS_LABEL.lower()
        ]

    texts: list[str] = []
    for _, row in labeled_df.iterrows():
        text = create_text_from_row(row.to_dict())
        if text:
            texts.append(text)

    logger.info("Loaded %d documents for indexing", len(texts))
    return texts, labeled_df
