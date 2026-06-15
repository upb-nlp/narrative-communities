"""Configuration for Microsoft GraphRAG indexing."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from narrative_map.config import (
    EMBED_API_BASE,
    EMBED_MODEL_NAME,
    GRAPHRAG_INPUT_DIR,
    GRAPHRAG_OUTPUT_DIR,
    GRAPHRAG_ROOT,
    LLM_API_BASE,
    LLM_MODEL_NAME,
    PROMPTS_DIR,
    REPO_ROOT,
    ensure_artifact_dirs,
)

logger = logging.getLogger(__name__)

MS_GRAPHRAG_ROOT = str(GRAPHRAG_ROOT)
MS_GRAPHRAG_INPUT_DIR = str(GRAPHRAG_INPUT_DIR)
MS_GRAPHRAG_OUTPUT_DIR = str(GRAPHRAG_OUTPUT_DIR)


def create_graphrag_project_structure() -> None:
    ensure_artifact_dirs()
    logger.info("GraphRAG project structure at %s", GRAPHRAG_ROOT)


def create_settings_yaml(
    target_communities: int = 20,
    resume_workflows: list | None = None,
) -> str:
    settings_path = GRAPHRAG_ROOT / "settings.yaml"

    llm_api_base = os.getenv("LLM_API_BASE", LLM_API_BASE)
    llm_model = os.getenv("LLM_MODEL", LLM_MODEL_NAME)
    embedding_api_base = os.getenv("EMBEDDING_API_BASE", EMBED_API_BASE)
    embedding_model = os.getenv("EMBEDDING_MODEL", EMBED_MODEL_NAME)

    settings = {
        "models": {
            "default_chat_model": {
                "type": "openai_chat",
                "api_base": llm_api_base,
                "model": llm_model,
                "api_key": "dummy",
                "encoding_model": "cl100k_base",
                "temperature": 0.3,
                "max_tokens": 2048,
                "concurrent_requests": 5,
                "requests_per_minute": 60,
                "request_timeout": 300.0,
                "max_retries": 3,
            },
            "default_embedding_model": {
                "type": "openai_embedding",
                "api_base": embedding_api_base,
                "model": embedding_model,
                "api_key": "dummy",
                "encoding_model": "cl100k_base",
                "dimensions": 4096,
                "concurrent_requests": 10,
                "request_timeout": 300.0,
                "max_retries": 10,
                "retry_strategy": "exponential_backoff",
            },
        },
        "parallelization": {"stagger": 0.5, "num_threads": 4},
        "storage": {"type": "local", "root": MS_GRAPHRAG_OUTPUT_DIR},
        "vector_store": {
            "default_vector_store": {
                "type": "lancedb",
                "db_uri": os.path.join(MS_GRAPHRAG_OUTPUT_DIR, "lancedb"),
                "container_name": "default",
                "overwrite": True,
            }
        },
        "input": {"local": {"base_dir": MS_GRAPHRAG_INPUT_DIR}},
        "chunks": {"size": 1000, "overlap": 200},
        "extract_graph": {
            "prompt": str(PROMPTS_DIR / "extract_graph.txt"),
            "max_gleanings": 2,
            "entity_types": [
                "PERSOANĂ",
                "ORGANIZAȚIE",
                "LOCAȚIE",
                "EVENIMENT",
                "BOALĂ",
                "MĂSURĂ DE SĂNĂTATE PUBLICĂ",
                "TRATAMENT",
                "VACCIN",
                "DEZINFORMARE",
            ],
        },
        "summarize_descriptions": {
            "prompt": str(PROMPTS_DIR / "summarize_descriptions.txt"),
            "max_length": 500,
        },
        "cluster_graph": {
            "max_cluster_size": 500,
            "use_lcc": True,
            "seed": 42,
        },
        "community_reports": {
            "graph_prompt": str(PROMPTS_DIR / "community_report_graph.txt"),
            "max_length": 1000,
            "max_input_length": 6000,
        },
    }

    if resume_workflows:
        settings["workflows"] = resume_workflows

    logger.info("GraphRAG settings: target_communities=~%d", target_communities)
    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.dump(settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("Created settings.yaml at %s", settings_path)
    return str(settings_path)


def create_env_file() -> str:
    env_path = GRAPHRAG_ROOT / ".env"
    env_path.write_text(
        "GRAPHRAG_API_KEY=dummy\nOPENAI_API_KEY=dummy\n",
        encoding="utf-8",
    )
    return str(env_path)


def initialize_config(
    target_communities: int = 20,
    resume_workflows: list | None = None,
) -> tuple[str, str]:
    create_graphrag_project_structure()
    settings_path = create_settings_yaml(
        target_communities=target_communities,
        resume_workflows=resume_workflows,
    )
    env_path = create_env_file()

    prompts_storage = GRAPHRAG_ROOT / "prompts"
    prompts_storage.mkdir(parents=True, exist_ok=True)
    logger.info("Prompts directory: %s", PROMPTS_DIR)
    return settings_path, env_path
