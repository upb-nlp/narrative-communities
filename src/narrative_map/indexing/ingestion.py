"""Microsoft GraphRAG ingestion pipeline."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from typing import List

import requests

from narrative_map.config import EMBED_API_BASE, EMBED_MODEL_NAME, LLM_API_URL, LLM_MODEL_NAME
from narrative_map.data import load_indexing_corpus
from narrative_map.indexing.config import (
    MS_GRAPHRAG_INPUT_DIR,
    MS_GRAPHRAG_ROOT,
    initialize_config,
)

logger = logging.getLogger(__name__)


def prepare_input_files(texts: List[str], output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = MS_GRAPHRAG_INPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Writing %d documents to %s…", len(texts), output_dir)

    for i, text in enumerate(texts):
        file_path = os.path.join(output_dir, f"document_{i:06d}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    logger.info("Prepared %d input files", len(texts))


def check_llm_api() -> bool:
    try:
        base_url = LLM_API_URL.rsplit("/v1", 1)[0]
        logger.info("Checking LLM server at %s…", base_url)
        try:
            requests.get(base_url, timeout=5)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to LLM server at %s", base_url)
            return False

        response = requests.post(
            LLM_API_URL,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": "test"}],
            },
            timeout=5,
        )
        if response.status_code in (200, 400, 401):
            logger.info("LLM API is accessible")
            return True
        if response.status_code == 404:
            logger.error("LLM API returned 404 for model %s", LLM_MODEL_NAME)
            return False
        logger.warning("LLM API returned status %d; continuing", response.status_code)
        return True
    except Exception as exc:
        logger.warning("LLM API check issue: %s", exc)
        return True


def check_embedding_api() -> bool:
    try:
        embedding_url = f"{EMBED_API_BASE.rstrip('/')}/embeddings"
        response = requests.post(
            embedding_url,
            json={"model": EMBED_MODEL_NAME, "input": ["test"]},
            timeout=5,
        )
        if response.status_code == 200:
            logger.info("Embedding API is accessible")
            return True
        logger.warning("Embedding API returned status %d", response.status_code)
        return False
    except requests.exceptions.RequestException as exc:
        logger.warning("Embedding API check failed: %s", exc)
        return False


def run_graphrag_index(root_dir: str | None = None) -> bool:
    if root_dir is None:
        root_dir = MS_GRAPHRAG_ROOT

    if not check_llm_api():
        logger.error("LLM API check failed; stopping before GraphRAG indexing")
        return False

    if not check_embedding_api():
        logger.warning("Embedding API check failed — ingestion may fail")

    cmd = ["graphrag", "index", "--root", root_dir, "--verbose"]
    logger.info("Starting GraphRAG indexing…")

    process = subprocess.Popen(
        cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)
        output_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    start_time = time.time()
    last_line_time = start_time
    stream_closed = False
    max_silent_seconds = int(os.getenv("GRAPHRAG_MAX_SILENT_SECONDS", "900"))

    while True:
        try:
            line = output_queue.get(timeout=5)
            if line is None:
                stream_closed = True
            else:
                line = line.rstrip()
                if line:
                    logger.info("[GraphRAG] %s", line)
                    output_lines.append(line)
                    last_line_time = time.time()
        except queue.Empty:
            pass

        if process.poll() is not None and stream_closed:
            break

        silent_for = int(time.time() - last_line_time)
        if max_silent_seconds > 0 and silent_for >= max_silent_seconds:
            logger.error("GraphRAG index stuck (no output for %ds); terminating", silent_for)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError("GraphRAG index timed out")

    returncode = process.wait()
    if returncode == 0:
        logger.info("GraphRAG indexing completed successfully")
        return True

    logger.error("GraphRAG indexing failed (exit code: %d)", returncode)
    return False


def initialize_microsoft_graphrag(resume_workflows: list | None = None) -> str:
    logger.info("Initializing Microsoft GraphRAG…")
    initialize_config(resume_workflows=resume_workflows)
    if resume_workflows:
        logger.info("Resume mode: workflows=%s", resume_workflows)
    logger.info("GraphRAG initialized at %s", MS_GRAPHRAG_ROOT)
    return MS_GRAPHRAG_ROOT


async def ingest_documents(texts: List[str], resume_workflows: list | None = None) -> None:
    logger.info("Ingesting %d documents…", len(texts))

    if not resume_workflows:
        prepare_input_files(texts)
    else:
        logger.info("Resume mode: skipping input preparation")

    if not run_graphrag_index():
        raise RuntimeError("Microsoft GraphRAG ingestion failed")

    logger.info("Document ingestion completed")


async def process_documents_once(resume_workflows: list | None = None) -> None:
    initialize_microsoft_graphrag(resume_workflows=resume_workflows)
    texts, _ = load_indexing_corpus(fake_only=True)
    logger.info("Loaded %d documents", len(texts))
    await ingest_documents(texts, resume_workflows=resume_workflows)
