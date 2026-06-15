"""CLI for DSPy community report generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from numbers import Number
from pathlib import Path

import pandas as pd

from narrative_map.community_reports.context_builder import load_community_contexts
from narrative_map.config import COMMUNITIES_DIR, REPORTS_DIR, REPO_ROOT

logger = logging.getLogger(__name__)


def _has_non_empty_title(report: dict) -> bool:
    title = report.get("title", "")
    return isinstance(title, str) and bool(title.strip())


def _is_valid_size(value: object) -> bool:
    return isinstance(value, Number)


def _select_top_n_by_size(contexts: list[dict], top_n: int) -> list[dict]:
    valid_contexts = [c for c in contexts if _is_valid_size(c.get("size"))]
    return sorted(valid_contexts, key=lambda x: x["size"], reverse=True)[:top_n]


def _preflight_dspy_import(timeout_sec: int = 90) -> None:
    os.environ.setdefault("LITELLM_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    probe = subprocess.run(
        [sys.executable, "-c", "import dspy; print('dspy_ok')"],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=os.environ.copy(),
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError("DSPy import failed. See logs above.")


def _get_report_val(report: dict | object, key: str, alt_key: str | None = None):
    if isinstance(report, dict):
        val = report.get(key)
        if val is None and alt_key:
            val = report.get(alt_key)
        return val if val is not None else (0 if key == "level" else "")
    val = getattr(report, key, None)
    if val is None and alt_key:
        val = getattr(report, alt_key, None)
    return val if val is not None else (0 if key == "level" else "")


def write_reports_parquet(reports: list, path: Path) -> None:
    new_rows = []
    for report in reports:
        cid = _get_report_val(report, "community_id")
        findings = _get_report_val(report, "findings_markdown", "findings")
        new_rows.append(
            {
                "id": str(cid),
                "human_readable_id": cid,
                "community": cid,
                "level": _get_report_val(report, "level") or 0,
                "title": _get_report_val(report, "title") or "",
                "summary": _get_report_val(report, "summary") or "",
                "full_content": _get_report_val(report, "full_content") or "",
                "text_for_embedding": _get_report_val(report, "text_for_embedding") or "",
                "rating_explanation": _get_report_val(report, "rating_explanation") or "",
                "findings": findings if isinstance(findings, str) else "",
            }
        )

    new_df = pd.DataFrame(new_rows)
    if path.exists():
        try:
            existing_df = pd.read_parquet(path)
            existing_df["community"] = existing_df["community"].astype(str)
            new_df["community"] = new_df["community"].astype(str)
            updated_ids = set(new_df["community"].tolist())
            existing_df = existing_df[~existing_df["community"].isin(updated_ids)]
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            final_df.to_parquet(path, index=False)
            logger.info("Updated parquet at %s (%d rows)", path, len(final_df))
            return
        except Exception as exc:
            logger.warning("Could not patch existing parquet: %s", exc)

    new_df.to_parquet(path, index=False)
    logger.info("Wrote parquet to %s (%d rows)", path, len(new_df))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate community reports using DSPy-enhanced pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with GraphRAG parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: artifacts/reports/<dirname>_dspy_community_reports.json)",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-entities", type=int, default=40)
    parser.add_argument("--max-relationships", type=int, default=60)
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write community_reports.parquet to data-dir",
    )
    parser.add_argument("--no-findings", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Retry communities missing from --existing-reports",
    )
    parser.add_argument(
        "--retry-empty-titles",
        action="store_true",
        help="Retry communities with empty titles in --existing-reports",
    )
    parser.add_argument("--existing-reports", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.retry_missing or args.retry_empty_titles:
        if not args.existing_reports:
            logger.error("--retry-missing/--retry-empty-titles require --existing-reports")
            sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
        force=True,
    )

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    elif (args.retry_missing or args.retry_empty_titles) and args.existing_reports:
        output_path = Path(args.existing_reports)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"{data_dir.name}_dspy_community_reports.json"

    contexts = load_community_contexts(
        data_dir,
        max_entities_per_community=args.max_entities,
        max_relationships_per_community=args.max_relationships,
    )
    if args.limit:
        contexts = contexts[: args.limit]
    elif args.top_n:
        contexts = _select_top_n_by_size(contexts, args.top_n)

    existing_reports_data: list[dict] = []
    if (args.retry_missing or args.retry_empty_titles) and args.existing_reports:
        existing_path = Path(args.existing_reports)
        if not existing_path.is_absolute():
            existing_path = REPO_ROOT / existing_path
        if not existing_path.exists():
            logger.error("Existing reports file not found: %s", existing_path)
            sys.exit(1)
        existing_reports_data = json.loads(existing_path.read_text(encoding="utf-8"))

        reported_ids = {r["community_id"] for r in existing_reports_data if "community_id" in r}
        all_ids = {c["community_id"] for c in contexts}
        missing_ids = all_ids - reported_ids
        empty_title_ids = {
            r["community_id"]
            for r in existing_reports_data
            if "community_id" in r and not _has_non_empty_title(r)
        }
        retry_ids: set = set()
        if args.retry_missing:
            retry_ids |= missing_ids
        if args.retry_empty_titles:
            retry_ids |= empty_title_ids & all_ids
        contexts = [c for c in contexts if c["community_id"] in retry_ids]
        if not contexts:
            logger.info("No matching communities to retry.")
            sys.exit(0)

    if not contexts:
        logger.warning("No communities found in %s", data_dir)
        sys.exit(0)

    _preflight_dspy_import()
    from narrative_map.community_reports.dspy_pipeline import generate_reports

    reports = asyncio.run(
        generate_reports(
            contexts,
            concurrency=args.concurrency,
            output_path=output_path,
            include_findings=not args.no_findings,
            existing_reports=existing_reports_data
            if (args.retry_missing or args.retry_empty_titles)
            else None,
        )
    )

    if args.write_parquet:
        write_reports_parquet(reports, data_dir / "community_reports.parquet")

    logger.info("Generated %d community reports", len(reports))


if __name__ == "__main__":
    main()
