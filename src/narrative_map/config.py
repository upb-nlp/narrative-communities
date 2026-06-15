"""Central configuration for the narrative-map pipeline."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtime artifacts (gitignored)
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
GRAPHRAG_ROOT = ARTIFACTS_DIR / "graphrag"
GRAPHRAG_INPUT_DIR = GRAPHRAG_ROOT / "input"
GRAPHRAG_OUTPUT_DIR = GRAPHRAG_ROOT / "output"
COMMUNITIES_DIR = ARTIFACTS_DIR / "communities"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Dataset
DATASET_NAME = "upb-nlp/ro_fake_news"
REAL_NEWS_LABEL = "Știre reală"

# Local LLM / embedding endpoints
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
EMBED_API_BASE = os.getenv("EMBEDDING_API_BASE", "http://localhost:8001/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen3.6-27B")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "qwen3-embedding")

LLM_API_URL = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
EMBEDDING_API_URL = f"{EMBED_API_BASE.rstrip('/')}/embeddings"

DEFAULT_LLM_URL = LLM_API_BASE
DEFAULT_EMBED_URL = EMBED_API_BASE

# Model artifact paths
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.pkl"
LIGHTGBM_SCALER_PATH = MODELS_DIR / "scaler.pkl"
DSPY_COMPILED_PATH = MODELS_DIR / "dspy_compiled.json"
SUMMARIES_CACHE_PATH = RESULTS_DIR / "summaries_cache.json"

PROMPTS_DIR = REPO_ROOT / "prompts"


def ensure_artifact_dirs() -> None:
    """Create artifact directories if they do not exist."""
    for path in (
        GRAPHRAG_ROOT,
        GRAPHRAG_INPUT_DIR,
        GRAPHRAG_OUTPUT_DIR,
        COMMUNITIES_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
