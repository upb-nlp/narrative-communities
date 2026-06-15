"""Narrative purity metrics (no plotting)."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from narrative_map.analysis.community_labels import (
    assign_primary_community,
    build_unique_community_documents,
    load_hf_text_to_supernarrative,
    parquet_path,
    PARQUET_TABLES,
)
from narrative_map.config import COMMUNITIES_DIR, REPORTS_DIR, RESULTS_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_GRAPHS: list[tuple[str, str, str | None]] = [
    ("combined", "new_all_qwen_36", "new_all_qwen_36_dspy_community_reports.json"),
    ("fake_only", "new_fake_news_36", "new_fake_news_36_dspy_community_reports.json"),
    ("real_only", "new_real_news_36", "new_real_news_36_dspy_community_reports.json"),
]


def purity_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return max(counts.values()) / total


def shannon_entropy_base2(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


def count_by_key(df: pd.DataFrame, label_col: str) -> dict[tuple, dict[str, int]]:
    out: dict[tuple, dict[str, int]] = {}
    labeled = df[df[label_col].notna()].copy()
    for (hid, title), grp in labeled.groupby(["human_readable_id", "title"], sort=False):
        counts = grp[label_col].value_counts().to_dict()
        out[(hid, title)] = {str(k): int(v) for k, v in counts.items()}
    return out


def is_narrative_blended(counts: dict[str, int], min_share: float) -> bool:
    total = sum(counts.values())
    if total <= 0:
        return False
    shares = [c / total for c in counts.values() if c > 0]
    return sum(1 for s in shares if s >= min_share) >= 2


def per_community_veracity_rows(
    counts_by_comm: dict[tuple, dict[str, int]],
    balance_min: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (hid, title), counts in counts_by_comm.items():
        n_fake = counts.get("fake", 0)
        n_real = counts.get("real", 0)
        n_docs = n_fake + n_real
        p_fake = n_fake / n_docs if n_docs else 0.0
        p_real = n_real / n_docs if n_docs else 0.0
        balanced = (
            n_docs > 0
            and n_fake > 0
            and n_real > 0
            and min(p_fake, p_real) >= balance_min
        )
        rows.append(
            {
                "human_readable_id": hid,
                "title": title,
                "n_docs": n_docs,
                "n_fake": n_fake,
                "n_real": n_real,
                "p_fake": round(p_fake, 4),
                "p_real": round(p_real, 4),
                "veracity_purity": round(purity_from_counts(counts), 4),
                "veracity_entropy": round(shannon_entropy_base2(counts), 4),
                "balanced_veracity_mix": balanced,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("n_docs", ascending=False).reset_index(drop=True) if not df.empty else df


def per_community_narrative_rows(
    counts_by_comm: dict[tuple, dict[str, int]],
    narrative_min_share: float,
) -> pd.DataFrame:
    all_labels: set[str] = set()
    for counts in counts_by_comm.values():
        all_labels.update(counts.keys())
    label_cols = sorted(all_labels)

    rows: list[dict[str, Any]] = []
    for (hid, title), counts in counts_by_comm.items():
        n_docs = sum(counts.values())
        row: dict[str, Any] = {
            "human_readable_id": hid,
            "title": title,
            "n_docs": n_docs,
            "narrative_purity": round(purity_from_counts(counts), 4),
            "narrative_entropy": round(shannon_entropy_base2(counts), 4),
            "narrative_blended": is_narrative_blended(counts, narrative_min_share),
            "n_distinct_narratives": len([c for c in counts.values() if c > 0]),
        }
        for lab in label_cols:
            row[f"n_{lab}"] = counts.get(lab, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("n_docs", ascending=False).reset_index(drop=True) if not df.empty else df


def pair_level_cluster_metrics(
    df: pd.DataFrame,
    label_col: str,
    counts_by_comm: dict[tuple, dict[str, int]],
) -> dict[str, Any]:
    labeled = df[df[label_col].notna()].copy()
    n_unknown = int(df[label_col].isna().sum())
    n_labeled = len(labeled)

    result: dict[str, Any] = {
        "n_pair_assignments_labeled": n_labeled,
        "n_pair_assignments_unknown": n_unknown,
        "n_communities_with_labeled_docs": len(counts_by_comm),
    }

    if n_labeled == 0 or not counts_by_comm:
        result.update(
            {
                "homogeneity": None,
                "completeness": None,
                "v_measure": None,
                "weighted_mean_purity": None,
                "macro_mean_purity": None,
            }
        )
        return result

    y_true = labeled[label_col].astype(str).tolist()
    y_pred = labeled["human_readable_id"].astype(str).tolist()
    n_classes = labeled[label_col].nunique()

    result["homogeneity"] = round(float(homogeneity_score(y_true, y_pred)), 4)
    if n_classes < 2:
        result["completeness"] = None
        result["v_measure"] = None
        result["metrics_note"] = "single_class"
    else:
        result["completeness"] = round(float(completeness_score(y_true, y_pred)), 4)
        result["v_measure"] = round(float(v_measure_score(y_true, y_pred)), 4)

    purities = [purity_from_counts(c) for c in counts_by_comm.values()]
    weights = [sum(c.values()) for c in counts_by_comm.values()]
    result["macro_mean_purity"] = round(float(np.mean(purities)), 4)
    result["weighted_mean_purity"] = round(float(np.average(purities, weights=weights)), 4)
    result["n_classes"] = int(n_classes)
    return result


def document_alignment_metrics(primary_df: pd.DataFrame, label_col: str) -> dict[str, Any]:
    labeled = primary_df[primary_df[label_col].notna()].copy()
    n_classes = int(labeled[label_col].nunique()) if len(labeled) else 0

    result: dict[str, Any] = {
        "n_documents": len(labeled),
        "n_classes": n_classes,
        "nmi": None,
        "ari": None,
    }

    if len(labeled) == 0:
        result["metrics_note"] = "no_labels"
        return result
    if n_classes < 2:
        result["metrics_note"] = "single_class"
        return result

    y_true = labeled[label_col].astype(str).tolist()
    y_pred = labeled["human_readable_id"].astype(str).tolist()
    result["nmi"] = round(float(normalized_mutual_info_score(y_true, y_pred)), 4)
    result["ari"] = round(float(adjusted_rand_score(y_true, y_pred)), 4)
    return result


def process_graph(
    graph_key: str,
    graph_dir: Path,
    communities_root: Path,
    reports_json: Path | None,
    hf_text_to_super: dict[str, str],
    *,
    balance_min: float,
    narrative_min_share: float,
) -> dict[str, Any]:
    for table in PARQUET_TABLES:
        if not parquet_path(graph_dir, table).exists():
            raise FileNotFoundError(f"{graph_key}: missing {parquet_path(graph_dir, table)}")

    long_df = build_unique_community_documents(
        graph_dir, communities_root, reports_json, hf_text_to_super
    )
    primary_df = assign_primary_community(long_df)

    narrative_counts = count_by_key(long_df, "narrative")
    veracity_counts = count_by_key(long_df, "veracity")

    sn_df = per_community_narrative_rows(narrative_counts, narrative_min_share)
    veracity_df = per_community_veracity_rows(veracity_counts, balance_min)

    narrative_pair = pair_level_cluster_metrics(long_df, "narrative", narrative_counts)
    veracity_pair = pair_level_cluster_metrics(long_df, "veracity", veracity_counts)
    narrative_align = document_alignment_metrics(primary_df, "narrative")
    veracity_align = document_alignment_metrics(primary_df, "veracity")

    n_multi = int((primary_df["n_communities_for_doc"] > 1).sum()) if len(primary_df) else 0

    return {
        "graph_key": graph_key,
        "graph_dir": str(graph_dir),
        "narrative": {**narrative_pair, "document_alignment": narrative_align},
        "veracity": {**veracity_pair, "document_alignment": veracity_align},
        "n_narrative_blended_communities": int(sn_df["narrative_blended"].sum())
        if not sn_df.empty
        else 0,
        "n_balanced_veracity_mix_communities": int(veracity_df["balanced_veracity_mix"].sum())
        if not veracity_df.empty
        else 0,
        "n_docs_primary_assignment": len(primary_df),
        "n_docs_multi_community": n_multi,
        "per_community_narrative": sn_df.to_dict(orient="records") if not sn_df.empty else [],
        "per_community_veracity": veracity_df.to_dict(orient="records") if not veracity_df.empty else [],
    }


def build_comparison_table(all_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in all_metrics:
        sn = m["narrative"]
        sn_align = sn.get("document_alignment", {})
        vr = m["veracity"]
        vr_align = vr.get("document_alignment", {})
        rows.append(
            {
                "graph_key": m["graph_key"],
                "graph_dir": Path(m["graph_dir"]).name,
                "sn_nmi": sn_align.get("nmi"),
                "sn_ari": sn_align.get("ari"),
                "sn_weighted_mean_purity": sn.get("weighted_mean_purity"),
                "n_narrative_blended": m["n_narrative_blended_communities"],
                "vr_nmi": vr_align.get("nmi"),
                "vr_ari": vr_align.get("ari"),
                "vr_weighted_mean_purity": vr.get("weighted_mean_purity"),
                "n_balanced_veracity_mix": m["n_balanced_veracity_mix_communities"],
                "n_docs_primary": m["n_docs_primary_assignment"],
                "n_docs_multi_community": m["n_docs_multi_community"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Single graph directory under artifacts/communities/",
    )
    parser.add_argument(
        "--communities-root",
        type=Path,
        default=COMMUNITIES_DIR,
    )
    parser.add_argument(
        "--reports-json",
        type=Path,
        default=None,
        help="DSPy community reports JSON for this graph",
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "purity")
    parser.add_argument("--balance-min", type=float, default=0.4)
    parser.add_argument("--narrative-min-share", type=float, default=0.15)
    parser.add_argument("--skip-hf", action="store_true")
    args = parser.parse_args()

    communities_root = args.communities_root
    if not communities_root.is_absolute():
        communities_root = REPO_ROOT / communities_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_map: dict[str, str] = {} if args.skip_hf else load_hf_text_to_supernarrative()

    if args.graph_dir:
        graph_dir = args.graph_dir if args.graph_dir.is_absolute() else REPO_ROOT / args.graph_dir
        reports_json = args.reports_json
        if reports_json and not reports_json.is_absolute():
            reports_json = REPO_ROOT / reports_json
        metrics = process_graph(
            graph_dir.name,
            graph_dir,
            communities_root,
            reports_json if reports_json and reports_json.exists() else None,
            hf_map,
            balance_min=args.balance_min,
            narrative_min_share=args.narrative_min_share,
        )
        all_metrics = [metrics]
    else:
        all_metrics = []
        for graph_key, rel_graph, rel_reports in DEFAULT_GRAPHS:
            graph_dir = communities_root / rel_graph
            if not graph_dir.exists():
                logger.warning("Skipping missing graph: %s", graph_dir)
                continue
            reports_json = REPORTS_DIR / rel_reports if rel_reports else None
            all_metrics.append(
                process_graph(
                    graph_key,
                    graph_dir,
                    communities_root,
                    reports_json if reports_json and reports_json.exists() else None,
                    hf_map,
                    balance_min=args.balance_min,
                    narrative_min_share=args.narrative_min_share,
                )
            )

    comparison = build_comparison_table(all_metrics)
    output = {"graphs": all_metrics, "comparison": comparison}
    out_path = out_dir / "global_metrics.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
