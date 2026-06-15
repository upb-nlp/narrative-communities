"""Shared helpers for community–document joins and document labels."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PARQUET_TABLES = ("communities", "text_units", "documents")

HF_BASE_URL = "hf://datasets/upb-nlp/ro_fake_news/"
HF_SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}

REAL_SUPER_LABELS = frozenset({"știre reală", "real news", "știre reala"})

SUPERNARRATIVE_MAPPING = {
    "acuzatii_de_autoritate_si_distopie": "Acuzații de autoritate și distopie",
    "criticarea_actorilor_nationali_si_internationali": "Criticarea actorilor naționali și internaționali",
    "criticarea_restrictiilor": "Criticarea restricțiilor",
    "geo_politica": "Geo-politică",
    "minimizarea_impactului_covid": "Minimizarea impactului COVID-19",
    "naratiuni_legate_de_sanatate": "Narațiuni legate de sănătate",
    "naratiuni_legate_de_vaccinare": "Narațiuni legate de vaccinare",
    "satira": "Satiră",
    "teorii_ale_conspiratiei": "Teorii ale conspirației",
}


def parquet_path(graph_dir: Path, table: str) -> Path:
    if table not in PARQUET_TABLES:
        raise ValueError(f"Unknown table {table!r}; use one of {PARQUET_TABLES}")
    return graph_dir / f"{table}.parquet"


def load_parquet_table(graph_dir: Path, table: str) -> pd.DataFrame:
    path = parquet_path(graph_dir, table)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_parquet(path)


def _hf_document_text(row: pd.Series) -> str:
    headline = str(row.get("headline") or "").strip()
    body = str(row.get("body") or "").strip()
    return f"{headline}\n\n{body}".strip()


def load_hf_text_to_supernarrative() -> dict[str, str]:
    """Map document text -> HF supernarrative (train + validation)."""
    logger.info("Loading HF dataset for supernarrative mapping...")
    text_to_label: dict[str, str] = {}
    for split, rel_path in HF_SPLITS.items():
        try:
            df = pd.read_parquet(HF_BASE_URL + rel_path)
            for _, row in df.iterrows():
                text = _hf_document_text(row)
                supernarrative = str(row.get("supernarrative") or "").strip()
                if text:
                    text_to_label[text] = supernarrative
        except Exception as e:
            logger.error("Error loading HF split %s: %s", split, e)
    logger.info("Loaded %s unique HF document mappings.", len(text_to_label))
    return text_to_label


def load_doc_to_super_folder(communities_root: Path) -> dict[Any, str]:
    """Legacy folder-based doc_id -> supernarrative display name."""
    doc_to_super: dict[Any, str] = {}
    for folder, name in SUPERNARRATIVE_MAPPING.items():
        doc_path = communities_root / folder / "documents.parquet"
        if doc_path.exists():
            df_docs = pd.read_parquet(doc_path)
            for doc_id in df_docs["id"].tolist():
                doc_to_super[doc_id] = name
    real_path = communities_root / "real" / "documents.parquet"
    if real_path.exists():
        df_real = pd.read_parquet(real_path)
        for doc_id in df_real["id"].tolist():
            if doc_id not in doc_to_super:
                doc_to_super[doc_id] = "Real News"
    return doc_to_super


def supernarrative_to_fake_real(supernarrative: str) -> str | None:
    if not supernarrative or supernarrative == "Unknown":
        return None
    if supernarrative.strip().lower() in REAL_SUPER_LABELS:
        return "real"
    return "fake"


def build_doc_id_labels(
    graph_dir: Path,
    communities_root: Path,
    hf_text_to_super: dict[str, str],
) -> tuple[dict[Any, str], dict[Any, str]]:
    documents_df = load_parquet_table(graph_dir, "documents")
    folder_super = load_doc_to_super_folder(communities_root)

    doc_to_narrative: dict[Any, str] = {}
    doc_to_fake_real: dict[Any, str] = {}

    for _, row in documents_df.iterrows():
        doc_id = row["id"]
        text = str(row.get("text") or "").strip()
        narrative: str | None = None

        if text and text in hf_text_to_super:
            narrative = hf_text_to_super[text]
        elif doc_id in folder_super:
            narrative = folder_super[doc_id]

        if narrative:
            doc_to_narrative[doc_id] = narrative
            fr = supernarrative_to_fake_real(narrative)
            if fr:
                doc_to_fake_real[doc_id] = fr

    return doc_to_narrative, doc_to_fake_real


def merge_community_titles(
    communities_df: pd.DataFrame,
    graph_dir: Path,
    reports_json: Path | None,
) -> pd.DataFrame:
    communities_df = communities_df.copy()
    reports_df = None
    if reports_json and reports_json.exists():
        reports_df = pd.read_json(reports_json)
    elif (graph_dir / "community_reports.parquet").exists():
        reports_df = pd.read_parquet(graph_dir / "community_reports.parquet")

    if reports_df is None:
        communities_df["title"] = "Community " + communities_df["human_readable_id"].astype(str)
        return communities_df

    reports_df = reports_df.copy()
    if "community_str" not in reports_df.columns:
        if "community_id" in reports_df.columns:
            reports_df["community_str"] = reports_df["community_id"].astype(str)
        elif "community" in reports_df.columns:
            reports_df["community_str"] = reports_df["community"].astype(str)

    communities_df["human_readable_id_str"] = communities_df["human_readable_id"].astype(str)
    merge_cols = [c for c in ("community_str", "title", "summary") if c in reports_df.columns]
    communities_df = pd.merge(
        communities_df,
        reports_df[merge_cols],
        left_on="human_readable_id_str",
        right_on="community_str",
        how="left",
        suffixes=("", "_report"),
    )
    if "title_report" in communities_df.columns:
        communities_df["title"] = communities_df["title_report"].fillna(
            "Community " + communities_df["human_readable_id_str"]
        )
    elif "title" not in communities_df.columns:
        communities_df["title"] = "Community " + communities_df["human_readable_id_str"]
    return communities_df


def build_unique_community_documents(
    graph_dir: Path,
    communities_root: Path,
    reports_json: Path | None,
    hf_text_to_super: dict[str, str],
) -> pd.DataFrame:
    communities_df = merge_community_titles(
        load_parquet_table(graph_dir, "communities"),
        graph_dir,
        reports_json,
    )
    text_units_df = load_parquet_table(graph_dir, "text_units")
    doc_to_narrative, doc_to_fake_real = build_doc_id_labels(
        graph_dir, communities_root, hf_text_to_super
    )

    communities_exploded = communities_df[
        ["id", "human_readable_id", "title", "text_unit_ids"]
    ].explode("text_unit_ids")
    text_units_subset = text_units_df[["id", "document_ids"]].rename(columns={"id": "text_unit_ids"})
    comm_to_docs = pd.merge(communities_exploded, text_units_subset, on="text_unit_ids")
    unique_comm_docs = comm_to_docs.explode("document_ids").drop_duplicates(
        subset=["id", "document_ids"]
    ).copy()

    unique_comm_docs = unique_comm_docs.rename(
        columns={"id": "community_uuid", "document_ids": "document_id"}
    )
    unique_comm_docs["narrative"] = unique_comm_docs["document_id"].map(doc_to_narrative)
    unique_comm_docs["veracity"] = unique_comm_docs["document_id"].map(doc_to_fake_real)
    return unique_comm_docs


def assign_primary_community(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per document: primary community (most text-unit links) plus labels."""
    pair_counts = (
        long_df.groupby(["document_id", "human_readable_id"], dropna=False)
        .size()
        .reset_index(name="n_links")
    )
    pair_counts["human_readable_id_sort"] = pair_counts["human_readable_id"].astype(str)
    pair_counts = pair_counts.sort_values(
        ["document_id", "n_links", "human_readable_id_sort"],
        ascending=[True, False, True],
    )
    primary = pair_counts.drop_duplicates("document_id", keep="first").drop(
        columns=["human_readable_id_sort"]
    )
    doc_labels = long_df.drop_duplicates("document_id")[
        ["document_id", "narrative", "veracity"]
    ]
    out = primary.merge(doc_labels, on="document_id", how="left")
    n_communities = long_df.groupby("document_id")["human_readable_id"].nunique()
    out["n_communities_for_doc"] = out["document_id"].map(n_communities)
    return out
