"""GraphRAG indexing CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from narrative_map.indexing.config import MS_GRAPHRAG_ROOT
from narrative_map.indexing.ingestion import process_documents_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_resume_workflows() -> list[str] | None:
    env_val = os.environ.get("GRAPHRAG_RESUME_WORKFLOWS", "").strip()
    if env_val:
        return [w.strip() for w in env_val.split(",") if w.strip()]
    return None


async def _find_last_failed_workflow() -> str | None:
    log_path = os.path.join(MS_GRAPHRAG_ROOT, "logs", "indexing-engine.log")
    if not os.path.exists(log_path):
        return None

    try:
        lines = open(log_path, encoding="utf-8").readlines()
        last_started = None
        completed_workflows: set[str] = set()
        for line in lines:
            if "Workflow started:" in line:
                last_started = line.split("Workflow started:")[-1].strip()
            if "Workflow completed:" in line:
                completed = line.split("Workflow completed:")[-1].strip()
                completed_workflows.add(completed)
        if last_started and last_started not in completed_workflows:
            return last_started
        return last_started
    except Exception as exc:
        logger.error("Error parsing log file: %s", exc)
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Microsoft GraphRAG indexing")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last failed workflow.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        metavar="WORKFLOW",
        help="Resume from a specific workflow.",
    )
    args = parser.parse_args()

    resume_workflows = None
    if args.resume:
        last_workflow = await _find_last_failed_workflow()
        if last_workflow:
            resume_workflows = [last_workflow]
            logger.info("Auto-resume from workflow: %s", resume_workflows)
    elif args.resume_from:
        resume_workflows = [args.resume_from]
    else:
        resume_workflows = _parse_resume_workflows()

    logger.info("Starting GraphRAG indexing")
    await process_documents_once(resume_workflows=resume_workflows)
    logger.info("Indexing completed. Project root: %s", MS_GRAPHRAG_ROOT)


if __name__ == "__main__":
    asyncio.run(main())
