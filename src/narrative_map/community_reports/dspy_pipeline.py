"""
DSPy-enhanced pipeline for generating community report titles and descriptions.

Order: **summary first** (from graph context), then **title** distilled from that
summary (plus context for names). Remaining steps unchanged (key phrases, etc.).
Optimized for prediction: descriptions are discriminative and embedding-rich.
Replaces GraphRAG's monolithic prompt with composable DSPy signatures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy

from narrative_map.llm.local_models import get_local_dspy_lm

from narrative_map.community_reports.title_postprocess import (
    strip_generic_community_title_prefix,
)

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """Write immediately to stderr (always visible)."""
    import sys
    sys.stderr.write(f"[dspy] {msg}\n")
    sys.stderr.flush()


def _has_non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _extract_findings_for_embedding(findings_markdown: str, max_chars: int = 320) -> str:
    """Extract a short findings fragment for retrieval embedding text."""
    if not findings_markdown or not isinstance(findings_markdown, str):
        return ""
    lines = [line.strip() for line in findings_markdown.splitlines() if line.strip()]
    kept: list[str] = []
    heading_count = 0
    for line in lines:
        if line.startswith("##"):
            heading_count += 1
            if heading_count > 2:
                break
        kept.append(line)
        if sum(len(x) for x in kept) >= max_chars:
            break
    fragment = " ".join(kept).strip()
    if len(fragment) > max_chars:
        fragment = fragment[: max_chars - 3].rstrip() + "..."
    return fragment


def _fallback_title_from_context(context: str, community_id: int | str) -> str:
    """Generate a deterministic context-grounded fallback title."""
    candidates: list[str] = []
    in_entities = False
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-----Entities-----"):
            in_entities = True
            continue
        if line.startswith("-----Relationships-----"):
            break
        if not in_entities:
            continue
        if line.lower().startswith("human_readable_id"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            candidates.append(parts[1])
        if len(candidates) >= 2:
            break
    if candidates:
        joined = " și ".join(candidates[:2])
        return f"{joined} în narativ"
    return f"Narativ comunitate {community_id}"

# ---------------------------------------------------------------------------
# DSPy Signatures (prediction-oriented)
# ---------------------------------------------------------------------------


class CommunityTitle(dspy.Signature):
    """Creează titlul DIN rezumatul analitic + context (pentru ortografierea numelor).

    Rezumatul stabilește deja unghiul narativ; titlul îl comprimă într-o linie.
    REGULI:
    - UNICITATE: Distilează din rezumat „ce e diferit” — nu inventa un alt subiect față de rezumat.
    - Folosește community_context doar pentru nume/locații corecte dacă lipsesc din rezumat.
    - INTERZIS: șabloane „Dinamica comunității”, „Rolul comunității”, „Analiza…”, enumerații goale „A și B”.
    - FLUIDITATE: prepoziții naturale. SPECIFICITATE: nume/loc/temă concretă.
    - Include explicit minimum DOUĂ diferențiatoare concrete: actori, mecanism, revendicare.
    - EXCLUSIV ROMÂNĂ. Maxim 12 cuvinte. FĂRĂ simboluri (**), FĂRĂ ":".
    """

    community_context = dspy.InputField(
        desc="Date structurate cu entități și locații (completare față de rezumat)."
    )
    summary = dspy.InputField(
        desc="Rezumatul analitic al comunității; sursa principală a unghiului din titlu."
    )
    title = dspy.OutputField(
        desc="Titlu fluid, corect gramatical și specific (max 12 cuvinte)."
    )


class CommunityDiscriminativeSummary(dspy.Signature):
    """Generează un rezumat dens și analitic EXCLUSIV în limba ROMÂNĂ.

    Rezumatul va fi folosit pentru matching: trebuie să fie MAXIM DISTINCTIV.

    Cerințe stricte:
    - EXCLUSIV ROMÂNĂ.
    - Începe direct cu esența narativă. Evită „Comunitatea se axează pe..." sau "Acest grup...".
    - Specifică clar TIPUL NARRATIV: conspirație, satiră, critică autoritate, factual, etc.
    - Include AFIRMAȚII SPECIFICE și ACTORI PRINCIPALI.
    - Include minimum DOUĂ diferențiatoare verificabile față de comunități similare:
      (actori unici, mecanism cauzal, tip de probă invocat, concluzie dominantă).
    - Fii extrem de CONCRET. Fără markdown sau simboluri decorative.
    Scrie în limba română, 180-250 cuvinte.
    """

    community_context = dspy.InputField(
        desc="Date structurate: entități, relații, afirmații ale comunității"
    )
    summary = dspy.OutputField(
        desc=(
            "Rezumat analitic distinctiv (180-250 cuvinte, română). Include: tip narativ, "
            "afirmații tipice, actori/surse, ton și diferențiatori expliciți. Fără markdown."
        )
    )


class CommunityKeyPhrases(dspy.Signature):
    """Extrage 5-8 fraze sau expresii reprezentative din comunitate.

    Aceste fraze vor fi folosite pentru îmbogățirea embedding-ului.
    Extrage afirmații caracteristice, sloganuri, nume de actori, termeni cheie.
    Fiecare frază: 3-12 cuvinte. Separate prin linie nouă.
    """

    community_context = dspy.InputField(
        desc="Date structurate ale comunității"
    )
    title = dspy.InputField(desc="Titlul comunității")
    summary = dspy.InputField(desc="Rezumatul comunității")
    key_phrases = dspy.OutputField(
        desc=(
            "5-8 fraze reprezentative, câte una pe linie. Fără numerotare. În limba română."
        )
    )


class CommunityFindingsMarkdown(dspy.Signature):
    """Generează 3-5 constatări în format Markdown (opțional).

    Fiecare constatare: ## TITLU\\n\\nEXPLICAȚIE.
    Nu returna JSON. Scrie direct în Markdown.
    """

    community_context = dspy.InputField(desc="Date structurate ale comunității")
    title = dspy.InputField(desc="Titlul comunității")
    summary = dspy.InputField(desc="Rezumatul comunității")
    findings_markdown = dspy.OutputField(
        desc=(
            "3-5 constatări în format Markdown: ## Titlu\\n\\nExplicație. "
            "Fără JSON, doar text Markdown."
        )
    )


class CommunityRating(dspy.Signature):
    """Evaluează relevanța comunității (0-10) pentru analiza dezinformării."""

    title = dspy.InputField(desc="Titlul comunității")
    summary = dspy.InputField(desc="Rezumatul comunității")
    rating = dspy.OutputField(desc="Float 0-10")
    rating_explanation = dspy.OutputField(desc="Explicație scurtă, română")


class RomanianRefiner(dspy.Signature):
    """Corector lingvistic ultra-strict.

    ROL: Transformă orice text brut în limba română literară, curată.
    REGULI ABSOLUTE:
    - DOAR ROMÂNĂ: Este STRICT INTERZISĂ folosirea caracterelor chinezești (ex: 感染者), chirilice sau a limbii engleze.
    - TRADUCERE: Dacă în textul sursă apar termeni străini, TRADUCE-I în română fluidă.
    - CORECTARE: Fixează "Carațină" -> "carantină", "Murgăți" -> "decedați".
    - DIACRITICE: Asigură folosirea corectă a literelor ă, î, â, ș, ț.
    - FĂRĂ SIMBOLURI: Elimină asteriscurile (**), hash-urile (#) sau liniile noi interne.
    """

    text_to_fix = dspy.InputField(desc="Text care poate conține caractere străine (chinezești/engleze) sau greșeli")
    fixed_text = dspy.OutputField(
        desc="Text pur în limba română, fără caractere străine, fără simboluri."
    )


class TitleConciseness(dspy.Signature):
    """Rafinează titlul pentru a fi concis, fluid și recognoscibil între comunități.

    Obiectiv: Maxim 10 cuvinte, ca un headline distinct — păstrează numele proprii și tema unică.
    Nu introduce și nu păstra prefixe generice („Dinamica comunității”, „Rolul…”, „Analiza…”).
    FĂRĂ simboluri, FĂRĂ două puncte, FĂRĂ Markdown.
    """

    raw_title = dspy.InputField(desc="Titlul brut generat anterior")
    concise_title = dspy.OutputField(
        desc="Titlu rafinat, jurnalistic, maxim 10 cuvinte, fără simboluri."
    )


class CommunityTitleAvoidingDuplicates(dspy.Signature):
    """Rescrie titlul pentru a evita coliziuni cu titluri similare din batch."""

    community_context = dspy.InputField(desc="Context comunitate pentru nume proprii corecte.")
    summary = dspy.InputField(desc="Rezumatul curent al comunității.")
    conflicting_titles = dspy.InputField(
        desc="Titluri similare de evitat; nu replica structura lor."
    )
    title = dspy.OutputField(
        desc=(
            "Titlu nou, clar diferențiat față de titlurile conflictuale. "
            "Maxim 12 cuvinte, română, fără markdown."
        )
    )


# ---------------------------------------------------------------------------
# Pipeline Module
# ---------------------------------------------------------------------------


class CommunityReportGenerator(dspy.Module):
    """Prediction-oriented pipeline: summary → title → refinement → conciseness → phrases → rating."""

    def __init__(self, include_findings: bool = True):
        super().__init__()
        self.generate_title = dspy.ChainOfThought(CommunityTitle)
        self.generate_unique_title = dspy.Predict(CommunityTitleAvoidingDuplicates)
        self.generate_summary = dspy.ChainOfThought(CommunityDiscriminativeSummary)
        self.refine = dspy.Predict(RomanianRefiner)
        self.make_concise = dspy.Predict(TitleConciseness)
        self.generate_key_phrases = dspy.Predict(CommunityKeyPhrases)
        self.generate_findings = (
            dspy.Predict(CommunityFindingsMarkdown) if include_findings else None
        )
        self.generate_rating = dspy.Predict(CommunityRating)

    def forward(self, community_context: str) -> dspy.Prediction:
        # Truncate context to avoid overwhelming the model (~8k chars ≈ 2k tokens)
        max_context_chars = 8000
        if len(community_context) > max_context_chars:
            orig_len = len(community_context)
            community_context = (
                community_context[:max_context_chars] + "\n\n[... truncated]"
            )
            logger.info(
                "Truncated context from %d to %d chars", orig_len, max_context_chars
            )

        _ns = 7 if self.generate_findings else 6
        _log(f"  Step 1/{_ns}: Generating summary...")
        summary_raw = self.generate_summary(community_context=community_context).summary

        _log(f"  Step 2/{_ns}: Refining summary (Romanian, strip symbols)...")
        summary = self.refine(text_to_fix=summary_raw).fixed_text

        _log(f"  Step 3/{_ns}: Generating title from summary...")
        title_raw = self.generate_title(
            community_context=community_context, summary=summary
        ).title

        _log(f"  Step 4/{_ns}: Refining title...")
        title_refined = self.refine(text_to_fix=title_raw).fixed_text

        _log(f"  Step 5/{_ns}: Making title concise...")
        title = self.make_concise(raw_title=title_refined).concise_title

        _log(f"  Step 6/{_ns}: Generating key phrases...")
        key_phrases_result = self.generate_key_phrases(
            community_context=community_context,
            title=title,
            summary=summary,
        )
        key_phrases = key_phrases_result.key_phrases

        findings_md = ""
        if self.generate_findings:
            _log(f"  Step {_ns}/{_ns}: Generating findings...")
            findings_result = self.generate_findings(
                community_context=community_context,
                title=title,
                summary=summary,
            )
            findings_md = findings_result.findings_markdown

        _log("  Generating rating...")
        rating_result = self.generate_rating(title=title, summary=summary)
        rating = rating_result.rating
        rating_explanation = rating_result.rating_explanation

        return dspy.Prediction(
            title=title,
            summary=summary,
            key_phrases=key_phrases,
            findings_markdown=findings_md,
            rating=rating,
            rating_explanation=rating_explanation,
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CommunityReport:
    """Structured result; includes text_for_embedding for prediction pipelines."""

    community_id: int | str
    level: int
    title: str = "Generating..."
    summary: str = ""
    key_phrases: str = ""
    findings_markdown: str = ""
    rating: float = 0.0
    rating_explanation: str = ""
    full_content: str = ""
    full_content_json: str = ""
    text_for_embedding: str = ""
    status: str = "pending"  # pending, generating, partial, complete, error

    def __post_init__(self):
        # Post-process to strip all asterisks and force single-line summaries
        def clean(t: str) -> str:
            if not t or not isinstance(t, str): return ""
            # Strip Chinese characters (range: 4E00-9FFF)
            t = re.sub(r'[\u4e00-\u9fff]+', '', t)
            # Strip bold symbols and newlines within blocks
            t = t.replace("**", "").replace("__", "")
            # Replace internal newlines with spaces for summary/title
            return " ".join(t.split())

        self.title = strip_generic_community_title_prefix(clean(self.title))
        self.summary = clean(self.summary)
        self.rating_explanation = clean(self.rating_explanation)

        # Update text for embedding and full content based on current fields
        parts = [self.title or "Untitled"]
        if self.summary:
            parts.append(self.summary)
        if self.key_phrases and self.key_phrases.strip():
            phrases = [p.strip() for p in self.key_phrases.strip().split("\n") if p.strip()]
            if phrases:
                parts.append("Fraze-cheie:\n" + "\n".join(phrases))
        findings_fragment = _extract_findings_for_embedding(self.findings_markdown)
        if findings_fragment:
            parts.append("Constatări:\n" + findings_fragment)
        self.text_for_embedding = "\n\n".join(parts)

        self.full_content = f"# {self.title or 'Untitled'}\n\n{self.summary or 'No summary yet.'}"
        if self.findings_markdown and self.findings_markdown.strip():
            self.full_content += f"\n\n{self.findings_markdown.strip()}"
        
        self.full_content_json = json.dumps(
            {
                "community_id": self.community_id,
                "status": self.status,
                "title": self.title,
                "summary": self.summary,
                "key_phrases": self.key_phrases.strip().split("\n") if self.key_phrases else [],
                "rating": self.rating,
                "rating_explanation": self.rating_explanation,
            },
            ensure_ascii=False,
            indent=4,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_key_phrases(raw: str) -> str:
    """Clean key phrases: remove numbering, empty lines, normalize."""
    if not raw or not isinstance(raw, str):
        return ""
    lines = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading "1. ", "2. ", "- ", "* "
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[\-\*]\s*", "", line)
        if line and len(line) >= 3:
            lines.append(line)
    return "\n".join(lines[:10])  # Cap at 10


def _parse_rating(raw: str) -> float:
    """Extract a float rating from potentially noisy LLM output."""
    if not raw: return 0.0
    raw = str(raw).strip().rstrip(".")
    try:
        return float(raw)
    except ValueError:
        pass
    # Try extracting first number
    import re
    match = re.search(r"(\d+\.?\d*)", raw)
    if match:
        return float(match.group(1))
    return 5.0


def _normalized_text_for_similarity(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-z0-9ăâîșț]+", (text or "").lower())
    return Counter(tokens)


def _counter_cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left.keys()) & set(right.keys())
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _detect_duplicate_pairs(
    reports: list[CommunityReport], threshold: float = 0.8
) -> list[tuple[int, int]]:
    vectors = [
        _normalized_text_for_similarity(f"{report.title} {report.summary}".strip())
        for report in reports
    ]
    pairs: list[tuple[int, int]] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = _counter_cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                pairs.append((i, j))
    return pairs


def _regenerate_colliding_titles(
    reports: list[CommunityReport],
    contexts_by_community_id: dict[int | str, str],
    generator: CommunityReportGenerator,
    max_rounds: int = 2,
) -> None:
    for _ in range(max_rounds):
        duplicate_pairs = _detect_duplicate_pairs(reports)
        if not duplicate_pairs:
            return
        conflicted_indexes: set[int] = set()
        for left_idx, right_idx in duplicate_pairs:
            conflicted_indexes.add(left_idx)
            conflicted_indexes.add(right_idx)
        for idx in sorted(conflicted_indexes):
            report = reports[idx]
            context = contexts_by_community_id.get(report.community_id, "")
            if not context:
                continue
            conflicting_titles = []
            for left_idx, right_idx in duplicate_pairs:
                if left_idx == idx:
                    conflicting_titles.append(reports[right_idx].title)
                elif right_idx == idx:
                    conflicting_titles.append(reports[left_idx].title)
            if not conflicting_titles:
                continue
            ctx = context[:8000] + "\n\n[... truncated]" if len(context) > 8000 else context
            rewritten = generator.generate_unique_title(
                community_context=ctx,
                summary=report.summary,
                conflicting_titles="\n".join(sorted(set(conflicting_titles))[:5]),
            ).title
            rewritten = str(generator.refine(text_to_fix=str(rewritten)).fixed_text).strip()
            rewritten = str(generator.make_concise(raw_title=rewritten).concise_title).strip()
            if _has_non_empty_text(rewritten):
                report.title = rewritten
            else:
                report.title = _fallback_title_from_context(ctx, report.community_id)
            report.__post_init__()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _configure_dspy():
    """Configure DSPy with the local LM if not already configured."""
    _log("Configuring DSPy with local LLM...")
    logger.info("Configuring DSPy with local LLM...")
    lm = get_local_dspy_lm(model_name="Qwen/Qwen3.6-35B-A3B")
    _log("Got LM, calling dspy.configure...")
    dspy.configure(lm=lm)
    _log("DSPy configured")


def generate_single_report(
    community_context: str,
    community_id: int | str,
    level: int,
    generator: CommunityReportGenerator,
    notify_callback: Any = None,
) -> CommunityReport:
    """Generate a community report for a single community, step-by-step."""
    
    report = CommunityReport(community_id=community_id, level=level, status="generating")
    if notify_callback: notify_callback(report)

    # Truncate context
    max_context_chars = 8000
    ctx = community_context
    if len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars] + "\n\n[... truncated]"

    try:
        _ns = 7 if generator.generate_findings else 6
        # Step 1: Summary (from context only)
        _log(f"  [{community_id}] Step 1/{_ns}: Summary...")
        res = generator.generate_summary(community_context=ctx)
        report.summary = str(res.summary).strip()
        report.__post_init__()
        if notify_callback: notify_callback(report)

        # Step 2: Refine summary
        _log(f"  [{community_id}] Step 2/{_ns}: Refine summary...")
        report.summary = str(generator.refine(text_to_fix=report.summary).fixed_text).strip()
        report.__post_init__()
        if notify_callback: notify_callback(report)

        # Step 3: Title from summary + context
        _log(f"  [{community_id}] Step 3/{_ns}: Title from summary...")
        res = generator.generate_title(community_context=ctx, summary=report.summary)
        report.title = str(res.title).strip()
        report.__post_init__()
        if notify_callback: notify_callback(report)

        # Step 4: Refine title
        _log(f"  [{community_id}] Step 4/{_ns}: Refine title...")
        report.title = str(generator.refine(text_to_fix=report.title).fixed_text).strip()
        report.__post_init__()
        if notify_callback: notify_callback(report)

        # Step 5: Concise title
        _log(f"  [{community_id}] Step 5/{_ns}: Conciseness...")
        report.title = str(generator.make_concise(raw_title=report.title).concise_title).strip()
        if not _has_non_empty_text(report.title):
            _log(
                f"  [{community_id}] Empty title after conciseness, retrying title from summary once..."
            )
            retry_title = str(
                generator.generate_title(
                    community_context=ctx, summary=report.summary
                ).title
            ).strip()
            retry_title = str(generator.refine(text_to_fix=retry_title).fixed_text).strip()
            retry_title = str(generator.make_concise(raw_title=retry_title).concise_title).strip()
            if _has_non_empty_text(retry_title):
                report.title = retry_title
            else:
                report.title = _fallback_title_from_context(ctx, community_id)
        report.__post_init__()
        if notify_callback: notify_callback(report)

        # Key phrases
        _log(f"  [{community_id}] Step 6/{_ns}: Key Phrases...")
        kp_res = generator.generate_key_phrases(community_context=ctx, title=report.title, summary=report.summary)
        report.key_phrases = _parse_key_phrases(str(kp_res.key_phrases))
        report.__post_init__()
        if notify_callback: notify_callback(report)

        if generator.generate_findings:
            _log(f"  [{community_id}] Step 7/{_ns}: Findings...")
            f_res = generator.generate_findings(community_context=ctx, title=report.title, summary=report.summary)
            report.findings_markdown = str(f_res.findings_markdown).strip()
            report.__post_init__()
            if notify_callback: notify_callback(report)

        # Step 7: Rating
        _log(f"  [{community_id}] Rating...")
        r_res = generator.generate_rating(title=report.title, summary=report.summary)
        report.rating = _parse_rating(r_res.rating)
        report.rating_explanation = str(r_res.rating_explanation).strip()
        report.status = "complete"
        report.__post_init__()
        if notify_callback: notify_callback(report)

    except Exception as e:
        _log(f"  [{community_id}] ERROR: {e}")
        logger.error("Error generating community %s: %s", community_id, e)
        report.status = "partial" if report.summary else "error"
        report.__post_init__()
        if notify_callback: notify_callback(report)

    return report


async def generate_reports(
    community_contexts: list[dict[str, Any]],
    concurrency: int = 3,
    output_path: str | Path | None = None,
    include_findings: bool = True,
    existing_reports: list[dict[str, Any]] | None = None,
) -> list[CommunityReport]:
    """Generate community reports for all communities using DSPy.

    Args:
        community_contexts: List of dicts from context_builder.load_community_contexts
        concurrency: Max parallel LLM calls
        output_path: Optional path to write results as JSON
        include_findings: If False, skip findings (faster, fewer LLM calls)
        existing_reports: Optional list of existing report dicts to merge with

    Returns:
        List of CommunityReport objects
    """
    _log("generate_reports() entered")
    _configure_dspy()
    _log("DSPy configured, creating generator")
    generator = CommunityReportGenerator(include_findings=include_findings)

    reports: list[CommunityReport] = []
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(max_workers=concurrency)
    _log(f"Created executor, spawning {len(community_contexts)} tasks...")

    # For iterative JSON saving, we keep a "working" list of dicts
    working_data_by_id = {}
    if existing_reports:
        for r in existing_reports:
            if "community_id" in r:
                working_data_by_id[r["community_id"]] = r

    jsonl_path = None
    if output_path:
        output_path = Path(output_path)
        jsonl_path = output_path.with_suffix(".jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _log(f"Iterative saving enabled: {output_path} (and .jsonl)")

    def _on_report_update(report: CommunityReport):
        """Callback to write current state to JSON."""
        if not output_path:
            return
        
        report_dict = {
            "community_id": report.community_id,
            "level": report.level,
            "status": report.status,
            "title": report.title,
            "summary": report.summary,
            "key_phrases": report.key_phrases,
            "text_for_embedding": report.text_for_embedding,
            "findings_markdown": report.findings_markdown,
            "rating": report.rating,
            "rating_explanation": report.rating_explanation,
            "full_content": report.full_content,
        }
        
        # We need thread-safe access to working_data_by_id if multiple workers call this,
        # but Python dict updates are atomic and we're just writing to a file.
        # However, writing the file itself should be protected to avoid truncation/corruption.
        # For simplicity in this script, we'll use a local lock.
        import threading
        if not hasattr(_on_report_update, "lock"):
            _on_report_update.lock = threading.Lock()
        
        with _on_report_update.lock:
            working_data_by_id[report.community_id] = report_dict
            sorted_ids = sorted(working_data_by_id.keys(), key=lambda x: (isinstance(x, str), x))
            serializable = [working_data_by_id[cid] for cid in sorted_ids]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)

    async def _process(ctx: dict[str, Any]) -> CommunityReport:
        async with semaphore:
            report = await loop.run_in_executor(
                executor,
                generate_single_report,
                ctx["context_string"],
                ctx["community_id"],
                ctx["level"],
                generator,
                _on_report_update if output_path else None,
            )
            if jsonl_path:
                report_dict = {
                    "community_id": report.community_id,
                    "level": report.level,
                    "title": report.title,
                    "summary": report.summary,
                    "key_phrases": report.key_phrases,
                    "text_for_embedding": report.text_for_embedding,
                    "findings_markdown": report.findings_markdown,
                    "rating": report.rating,
                    "rating_explanation": report.rating_explanation,
                    "full_content": report.full_content,
                }
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(report_dict, ensure_ascii=False) + "\n")
            logger.info(
                "Generated report for community %s: %s",
                ctx["community_id"],
                report.title[:80],
            )
            return report

    tasks = [asyncio.create_task(_process(ctx)) for ctx in community_contexts]

    for coro in asyncio.as_completed(tasks):
        try:
            report = await coro
            reports.append(report)
        except Exception:
            logger.exception("Failed to generate report for a community")

    executor.shutdown(wait=False)
    reports.sort(key=lambda r: r.community_id)
    contexts_by_community_id = {
        ctx["community_id"]: ctx["context_string"] for ctx in community_contexts
    }
    _regenerate_colliding_titles(
        reports=reports,
        contexts_by_community_id=contexts_by_community_id,
        generator=generator,
    )
    return reports
