"""
Build per-community context strings from GraphRAG parquet artifacts.

Reads entities.parquet, relationships.parquet, and communities.parquet
to reconstruct the CSV-block context that GraphRAG feeds to its community
report prompt. This lets us regenerate reports with a DSPy pipeline without
re-running the full GraphRAG indexing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    sys.stderr.write(f"[context] {msg}\n")
    sys.stderr.flush()


def _entity_community_map(communities_df: pd.DataFrame) -> dict[str, list[int]]:
    """Map entity IDs to their community IDs.

    The communities parquet stores entity_ids as an array column. We invert
    that into a dict keyed by entity id.
    """
    entity_to_communities: dict[str, list[int]] = {}
    for _, row in communities_df.iterrows():
        comm_id = row.get("community") or row.get("id")
        entity_ids = row.get("entity_ids", [])
        if entity_ids is None:
            continue
        for eid in entity_ids:
            entity_to_communities.setdefault(str(eid), []).append(comm_id)
    return entity_to_communities


def _build_entity_block(entities: pd.DataFrame) -> str:
    """Format entities into the CSV block GraphRAG uses."""
    if entities.empty:
        return ""
    cols = ["human_readable_id", "title", "description", "degree"]
    available = [c for c in cols if c in entities.columns]
    return f"-----Entities-----\n{entities[available].to_csv(index=False, sep=',')}"


def _build_relationship_block(relationships: pd.DataFrame) -> str:
    """Format relationships into the CSV block GraphRAG uses."""
    if relationships.empty:
        return ""
    cols = ["human_readable_id", "source", "target", "description", "combined_degree"]
    available = [c for c in cols if c in relationships.columns]
    if "combined_degree" not in relationships.columns and "weight" in relationships.columns:
        relationships = relationships.copy()
        relationships["combined_degree"] = relationships["weight"]
        available = [c for c in cols if c in relationships.columns]
    return f"-----Relationships-----\n{relationships[available].to_csv(index=False, sep=',')}"


def _select_with_diversity(
    df: pd.DataFrame,
    max_rows: int,
    sort_column: str | None,
    diversity_rows: int = 8,
) -> pd.DataFrame:
    """Keep high-salience rows and add a deterministic diversity slice from the tail."""
    if df.empty or max_rows <= 0:
        return df.head(0)
    ranked = df
    if sort_column and sort_column in ranked.columns:
        ranked = ranked.sort_values(sort_column, ascending=False)
    if len(ranked) <= max_rows:
        return ranked

    diversity_rows = max(0, min(diversity_rows, max_rows // 3))
    core_rows = max_rows - diversity_rows
    top = ranked.head(core_rows)
    if diversity_rows == 0:
        return top

    tail = ranked.iloc[core_rows:]
    if tail.empty:
        return top
    stride = max(1, len(tail) // diversity_rows)
    sampled = tail.iloc[::stride].head(diversity_rows)
    selected = pd.concat([top, sampled], axis=0)
    return selected.loc[~selected.index.duplicated(keep="first")].head(max_rows)


def load_community_contexts(
    data_dir: str | Path,
    max_entities_per_community: int = 40,
    max_relationships_per_community: int = 60,
) -> list[dict[str, Any]]:
    """Load parquet files and build context strings for every community.

    Returns a list of dicts, each with keys:
        community_id, level, size, context_string,
        and optionally the original title/summary from existing reports.
    """
    data_dir = Path(data_dir)
    _log(f"load_community_contexts: reading parquet from {data_dir}")

    entities_df = pd.read_parquet(data_dir / "entities.parquet")
    relationships_df = pd.read_parquet(data_dir / "relationships.parquet")
    communities_df = pd.read_parquet(data_dir / "communities.parquet")

    # Optionally load existing reports to compare later
    existing_reports: dict[Any, dict] = {}
    reports_path = data_dir / "community_reports.parquet"
    if reports_path.exists():
        reports_df = pd.read_parquet(reports_path)
        for _, row in reports_df.iterrows():
            existing_reports[row["community"]] = {
                "title": row.get("title", ""),
                "summary": row.get("summary", ""),
            }

    entity_to_comms = _entity_community_map(communities_df)

    # Index entities by their id for fast lookup
    entity_id_col = "id"
    entity_index = entities_df.set_index(entity_id_col)

    contexts: list[dict[str, Any]] = []

    for _, comm_row in communities_df.iterrows():
        comm_id = comm_row.get("community", comm_row.get("id"))
        level = comm_row.get("level", 0)
        entity_ids = comm_row.get("entity_ids", [])
        if entity_ids is None:
            entity_ids = []
        entity_ids = [str(eid) for eid in entity_ids]

        # Gather entities for this community
        matching_entities = entity_index[entity_index.index.isin(entity_ids)]
        if matching_entities.empty:
            continue

        matching_entities = _select_with_diversity(
            matching_entities,
            max_rows=max_entities_per_community,
            sort_column="degree",
            diversity_rows=8,
        )

        # Gather relationships where both source & target are in this community
        entity_titles = set(matching_entities["title"].tolist()) if "title" in matching_entities.columns else set()
        matching_rels = relationships_df[
            relationships_df["source"].isin(entity_titles)
            & relationships_df["target"].isin(entity_titles)
        ]
        rel_sort_col = "weight" if "weight" in matching_rels.columns else "combined_degree"
        matching_rels = _select_with_diversity(
            matching_rels,
            max_rows=max_relationships_per_community,
            sort_column=rel_sort_col,
            diversity_rows=10,
        )

        entity_block = _build_entity_block(matching_entities.reset_index())
        rel_block = _build_relationship_block(matching_rels)

        context_string = "\n\n".join(b for b in [entity_block, rel_block] if b)

        existing = existing_reports.get(comm_id, {})

        contexts.append({
            "community_id": comm_id,
            "level": level,
            "size": len(entity_ids),
            "context_string": context_string,
            "original_title": existing.get("title", ""),
            "original_summary": existing.get("summary", ""),
        })

    _log(f"load_community_contexts: built {len(contexts)} contexts")
    logger.info("Built context for %d communities from %s", len(contexts), data_dir)
    return contexts
