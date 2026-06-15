"""
Fake news prediction pipeline.

LightGBM  →  community-similarity  →  DSPy judge  →  ensemble.

1.  LightGBM gives a fast first prediction + confidence score.
2.  A **DSPy** article summary (``ArticleSummarization``) is built for
    **community retrieval** (cached in ``summaries_cache.json`` in batch
    runs). The summary text is embedded — not the raw body — so it matches
    the granularity of community **summary** fields. Falls back to
    headline+truncated body only if no LLM is available.
3.  A DSPy-based classifier (structured I/O, community patterns)
    independently classifies the article — without seeing the ML
    prediction, to avoid anchoring bias.
4.  Ensemble (``linear_v1``): weighted mix of LightGBM + DSPy + community signals
    (see ``narrative_map.prediction.ensemble``; tune ``--linear-threshold`` and ``--linear-w``).

"""
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from narrative_map.prediction.classifier import LightGBMClassifier
from narrative_map.config import (
    DEFAULT_EMBED_URL,
    DEFAULT_LLM_URL,
    RESULTS_DIR as ARTIFACT_RESULTS_DIR,
    SUMMARIES_CACHE_PATH,
)
from narrative_map.data import (
    DATASET_NAME,
    DatasetSplit,
    ground_truth_label,
    load_dataset_split,
)
from narrative_map.prediction.ensemble import (
    DecisionContext,
    DecisionOutcome,
    ensemble_decide,
    make_decision_context,
    normalize_conf,
    normalize_label,
)

DEFAULT_TOP_K = 5
RESULTS_DIR = ARTIFACT_RESULTS_DIR


def results_path_for_split(split: DatasetSplit) -> Path:
    """Split-aware CSV export path."""
    ARTIFACT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_RESULTS_DIR / f"prediction_{split}.csv"


def failures_path_for_split(split: DatasetSplit) -> Path:
    ARTIFACT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_RESULTS_DIR / f"failed_predictions_{split}.md"


# ------------------------------------------------------------------
# Summary cache
# ------------------------------------------------------------------
def _article_key(headline: str, body: str) -> str:
    """Deterministic short hash for an article."""
    text = f"{headline}\n{body}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class SummaryCache:
    """Persistent JSON cache for LLM-generated article summaries."""

    def __init__(self, path: Path = SUMMARIES_CACHE_PATH):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            print(f"  Loaded {len(self.data)} cached summaries from {path.name}")

    def get(self, headline: str, body: str) -> str | None:
        return self.data.get(_article_key(headline, body))

    def put(self, headline: str, body: str, summary: str):
        self.data[_article_key(headline, body)] = summary

    def save(self):
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


async def summarize_article(
    headline: str,
    body: str,
    base_url: str,
    model: str,
) -> str:
    """Call the LLM to produce a retrieval-oriented summary via DSPy."""
    from narrative_map.prediction.dspy_judge import (
        configure_dspy_lm, DSPySummarizer,
    )

    configure_dspy_lm(base_url=base_url, model=model)
    summarizer = DSPySummarizer(use_cot=False)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: summarizer(headline=headline, body=body),
        )
        return (result.summary or "").strip()
    except Exception as exc:
        print(f"  ⚠ DSPy Summary error: {exc}")
        return f"{headline} {body[:500]}"


async def resolve_doc_summary_for_retrieval(
    headline: str,
    body: str,
    *,
    explicit_summary: str | None,
    summary_cache: SummaryCache | None,
    summary_llm_url: str | None,
    summary_model: str | None,
) -> str:
    """DSPy summary for community cosine similarity (aligned with community summaries)."""
    if explicit_summary and str(explicit_summary).strip():
        return str(explicit_summary).strip()
    if summary_cache:
        hit = summary_cache.get(headline, body)
        if hit and hit.strip():
            return hit.strip()
    if summary_llm_url and summary_model:
        from narrative_map.prediction.dspy_judge import configure_dspy_lm

        configure_dspy_lm(base_url=summary_llm_url, model=summary_model)
        return await summarize_article(
            headline, body, summary_llm_url, summary_model,
        )
    return ""


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Result of a single prediction."""

    headline: str
    prediction: str
    confidence: float
    reasoning: str
    ground_truth: str
    is_correct: bool
    sample_id: str = ""
    split: str = ""
    lgbm_pred: str = ""
    lgbm_conf: float = 0.0
    llm_decision: str = ""
    llm_conf: float = 0.0
    llm_explanation: str | None = None
    matched_community: str = ""
    community_explanation: str = ""
    community_stats: dict = field(default_factory=dict)
    top_communities: dict = field(default_factory=dict)
    doc_summary: str = ""


@dataclass
class JudgeDecision:
    """Normalized judge output before ensemble policy selection."""

    source: str
    label: str = ""
    confidence: float = 0.0
    explanation: str = ""
    matched_community: str = ""
    community_explanation: str = ""
    error: str = ""


# ------------------------------------------------------------------
# Predictor
# ------------------------------------------------------------------
class GraphRAGPredictor:
    """LightGBM + community-similarity + DSPy judge predictor."""

    def __init__(
        self,
        lgbm_classifier: LightGBMClassifier | None = None,
        community_sim=None,
        top_k: int = DEFAULT_TOP_K,
        dspy_judge=None,
        linear_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        linear_threshold: float = 0.5,
        summary_cache: SummaryCache | None = None,
        summary_llm_url: str | None = None,
        summary_model: str | None = None,
        **_kw,
    ):
        self.lgbm_classifier = lgbm_classifier
        self.community_sim = community_sim
        self.top_k = top_k
        self.dspy_judge = dspy_judge
        self.linear_weights = linear_weights
        self.linear_threshold = linear_threshold
        self.summary_cache = summary_cache
        self.summary_llm_url = summary_llm_url
        self.summary_model = summary_model

    def _build_decision_context(
        self,
        *,
        lgbm_pred: str,
        lgbm_conf: float,
        llm_pred: str,
        llm_conf: float,
        community_result: dict | None,
        source: str,
    ) -> DecisionContext:
        return make_decision_context(
            lgbm_pred=lgbm_pred,
            lgbm_conf=lgbm_conf,
            llm_pred=llm_pred,
            llm_conf=llm_conf,
            sim_stats=community_result["stats"] if community_result else {},
            source=source,
            has_judge=bool(llm_pred),
        )

    def _run_policy(self, context: DecisionContext) -> DecisionOutcome:
        return ensemble_decide(
            context,
            linear_weights=self.linear_weights,
            linear_threshold=self.linear_threshold,
        )

    @staticmethod
    def _normalize_judge_decision(
        *,
        source: str,
        raw_label: str,
        raw_conf: float,
        explanation: str | None = None,
        matched_community: str | None = None,
        community_explanation: str | None = None,
        error: str = "",
    ) -> JudgeDecision:
        label = normalize_label(raw_label, fallback="")
        conf = normalize_conf(raw_conf) if label else 0.0
        return JudgeDecision(
            source=source,
            label=label,
            confidence=conf,
            explanation=(explanation or "").strip(),
            matched_community=(matched_community or "").strip(),
            community_explanation=(community_explanation or "").strip(),
            error=error.strip(),
        )

    def _format_reasoning(
        self,
        *,
        parts: list[str],
        decision: JudgeDecision,
        outcome: DecisionOutcome,
        lgbm_conf: float,
    ) -> None:
        if decision.error:
            parts.append(f"{decision.source} error: {decision.error[:100]}")
            return
        if not decision.label:
            return
        src = "DSPy"
        short_expl = decision.explanation[:120] if decision.explanation else ""
        if outcome.did_override and outcome.label == decision.label:
            parts.append(
                f"{src} override → {decision.label.upper()} "
                f"({decision.confidence:.0%})"
                + (f" — {short_expl}" if short_expl else "")
            )
        elif decision.label == outcome.debug.get("lgbm_pred", ""):
            parts.append(
                f"{src} agrees: {decision.label.upper()} "
                f"({decision.confidence:.0%})"
                + (f" — {short_expl}" if short_expl else "")
            )
        else:
            parts.append(
                f"{src} suggested {decision.label.upper()} "
                f"({decision.confidence:.0%}) but kept LightGBM "
                f"(lgbm={lgbm_conf:.0%})"
            )
        if decision.matched_community:
            parts.append(f"Matched community: {decision.matched_community}")

    # ---- single article --------------------------------------------------

    async def predict(
        self,
        headline: str,
        body: str,
        summary: str | None = None,
    ) -> PredictionResult:
        if not self.lgbm_classifier or not self.lgbm_classifier.is_trained:
            return PredictionResult(
                headline=headline[:100],
                prediction="fake",
                confidence=0.5,
                reasoning="LightGBM classifier not available",
                ground_truth="",
                is_correct=False,
            )

        try:
            doc_summary = ""
            # --- Step 1: LightGBM ----------------------------------------
            fs = self.lgbm_classifier.extract_features(headline, body)
            lgbm_pred, lgbm_conf = self.lgbm_classifier.predict(
                fs, headline, body
            )

            parts = [f"LightGBM: {lgbm_pred.upper()} ({lgbm_conf:.0%})"]

            # --- Step 2: community similarity (embed DSPy body summary) ---
            community_result = None
            if self.community_sim is not None:
                doc_summary = await resolve_doc_summary_for_retrieval(
                    headline,
                    body,
                    explicit_summary=summary,
                    summary_cache=self.summary_cache,
                    summary_llm_url=self.summary_llm_url,
                    summary_model=self.summary_model,
                )
                if self.summary_cache and doc_summary:
                    if self.summary_cache.get(headline, body) != doc_summary:
                        self.summary_cache.put(headline, body, doc_summary)
                if doc_summary.strip():
                    # Match community side: community vectors are built from "summary" text
                    query_text = doc_summary.strip()
                else:
                    query_text = (
                        f"{headline} {body[:2000]}" if body else headline
                    )
                community_result = self.community_sim.query(
                    query_text, top_k=self.top_k
                )
                stats = community_result["stats"]
                parts.append(
                    f"Sim fake_max={stats['fake_max_sim']:.3f}, "
                    f"real_max={stats['real_max_sim']:.3f}"
                )

            # --- Step 3: DSPy / LLM judge ---------------------------------
            final_pred = lgbm_pred
            final_conf = lgbm_conf
            llm_expl = None

            judge_decision = JudgeDecision(source="none")
            outcome = self._run_policy(
                self._build_decision_context(
                    lgbm_pred=lgbm_pred,
                    lgbm_conf=lgbm_conf,
                    llm_pred="",
                    llm_conf=0.0,
                    community_result=community_result,
                    source="lgbm_only",
                )
            )

            matched_community = ""
            community_explanation = ""
            llm_decision = ""
            llm_conf = 0.0

            if self.dspy_judge is not None:
                try:
                    judge_result = await self.dspy_judge.judge(
                        headline=headline, body=body,
                        top_fake=(
                            community_result["top_fake"]
                            if community_result else None
                        ),
                        top_real=(
                            community_result["top_real"]
                            if community_result else None
                        ),
                        stats=(
                            community_result["stats"]
                            if community_result else None
                        ),
                        top_k=self.top_k,
                    )
                    judge_decision = self._normalize_judge_decision(
                        source="dspy",
                        raw_label=judge_result.decision,
                        raw_conf=judge_result.confidence,
                        explanation=judge_result.explanation or judge_result.reasoning,
                        matched_community=judge_result.matched_community,
                        community_explanation=judge_result.explanation,
                    )
                    context = self._build_decision_context(
                        lgbm_pred=lgbm_pred,
                        lgbm_conf=lgbm_conf,
                        llm_pred=judge_decision.label,
                        llm_conf=judge_decision.confidence,
                        community_result=community_result,
                        source="dspy",
                    )
                    outcome = self._run_policy(context)
                except Exception as exc:
                    judge_decision = JudgeDecision(source="dspy", error=str(exc))

            final_pred = outcome.label
            final_conf = outcome.confidence
            llm_decision = judge_decision.label
            llm_conf = judge_decision.confidence
            llm_expl = judge_decision.explanation or None
            matched_community = judge_decision.matched_community
            community_explanation = judge_decision.community_explanation
            self._format_reasoning(
                parts=parts,
                decision=judge_decision,
                outcome=outcome,
                lgbm_conf=lgbm_conf,
            )

            return PredictionResult(
                headline=(
                    headline[:100] + "…"
                    if len(headline) > 100
                    else headline
                ),
                prediction=final_pred,
                confidence=final_conf,
                reasoning=". ".join(parts),
                ground_truth="",
                is_correct=False,
                lgbm_pred=lgbm_pred,
                lgbm_conf=lgbm_conf,
                llm_decision=llm_decision,
                llm_conf=llm_conf,
                llm_explanation=llm_expl,
                matched_community=matched_community,
                community_explanation=community_explanation,
                community_stats=(
                    community_result["stats"] if community_result else {}
                ),
                top_communities={
                    "top_fake": community_result["top_fake"],
                    "top_real": community_result["top_real"],
                } if community_result else {},
                doc_summary=doc_summary or (summary or ""),
            )

        except Exception as exc:
            return PredictionResult(
                headline=headline[:100],
                prediction="fake",
                confidence=0.5,
                reasoning=f"Error: {str(exc)[:120]}",
                ground_truth="",
                is_correct=False,
            )


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_test_data(
    limit: int | None = None, balanced: bool = False
) -> list[dict]:
    """Load test split from the dataset (backward-compatible alias)."""
    return load_dataset_split("test", limit=limit, balanced=balanced)


def get_ground_truth(sample: dict) -> str:
    return ground_truth_label(sample)


# ------------------------------------------------------------------
# Pre-compute summaries
# ------------------------------------------------------------------
async def precompute_summaries(
    test_data: list[dict],
    llm_base_url: str,
    llm_model: str,
    cache: SummaryCache,
) -> SummaryCache:
    """Summarise all articles that aren't already cached, then save."""
    missing = [
        s for s in test_data
        if cache.get(s.get("headline", ""), s.get("body", "")) is None
    ]
    if not missing:
        print(f"  All {len(test_data)} summaries already cached ✓")
        return cache

    print(
        f"  {len(missing)} articles need summarisation "
        f"({len(test_data) - len(missing)} cached)"
    )
    for sample in tqdm(missing, desc="Summarising"):
        headline = sample.get("headline", "")
        body = sample.get("body", "")
        try:
            summary = await summarize_article(
                headline, body, llm_base_url, llm_model,
            )
        except Exception as exc:
            summary = f"{headline} {body[:500]}"
            print(f"  ⚠ Summary fallback for «{headline[:60]}»: {exc}")
        cache.put(headline, body, summary)

    cache.save()
    print(f"  ✓ Saved {len(cache.data)} summaries to {cache.path.name}")
    return cache


# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------
async def run_predictions(
    predictor: GraphRAGPredictor,
    test_data: list[dict],
    summary_cache: SummaryCache | None = None,
    verbose: bool = True,
    *,
    split: DatasetSplit = "test",
) -> list[PredictionResult]:
    results: list[PredictionResult] = []

    for i, sample in enumerate(tqdm(test_data, desc="Predicting")):
        headline = sample.get("headline", "")
        body = sample.get("body", "")
        ground_truth = get_ground_truth(sample)
        sample_id = str(sample.get("id", ""))

        summary = (
            summary_cache.get(headline, body)
            if summary_cache
            else None
        )

        if verbose and (i + 1) % 10 == 0:
            print(f"\n  Processing sample {i + 1}/{len(test_data)}")

        result = await predictor.predict(headline, body, summary=summary)
        result.ground_truth = ground_truth
        result.is_correct = result.prediction == ground_truth
        result.sample_id = sample_id
        result.split = split
        results.append(result)

        if verbose and not result.is_correct:
            print(
                f"  ✗ Misclassified: predicted={result.prediction}, "
                f"actual={ground_truth}"
            )

    # ---- Diagnostics: override stats ---------------------------------
    agree = sum(1 for r in results if "DSPy agrees" in r.reasoning)
    override = sum(1 for r in results if "DSPy override" in r.reasoning)
    kept = sum(1 for r in results if "but kept LightGBM" in r.reasoning)
    judge_err = sum(1 for r in results if "DSPy error" in r.reasoning)
    no_judge = len(results) - agree - override - kept - judge_err
    print("\n  Judge diagnostics:")
    print(f"    Agree with LightGBM : {agree}")
    print(f"    Override accepted   : {override}")
    print(f"    Override blocked    : {kept}")
    print(f"    Judge errors        : {judge_err}")
    print(f"    No judge (fallback) : {no_judge}")
    if override + kept > 0:
        override_correct = sum(
            1 for r in results
            if "DSPy override" in r.reasoning and r.is_correct
        )
        kept_would_flip = sum(
            1 for r in results
            if "but kept LightGBM" in r.reasoning and not r.is_correct
        )
        print(f"    Overrides correct    : {override_correct}/{override}")
        print(f"    Blocked but LightGBM wrong: {kept_would_flip}/{kept}")

    return results


def compute_metrics(results: list[PredictionResult]) -> dict:
    y_true = [1 if r.ground_truth == "real" else 0 for r in results]
    y_pred = [1 if r.prediction == "real" else 0 for r in results]
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy_score(y_true, y_pred) if total else 0.0,
        "precision": (
            precision_score(y_true, y_pred, zero_division=0) if total else 0.0
        ),
        "recall": (
            recall_score(y_true, y_pred, zero_division=0) if total else 0.0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0) if total else 0.0,
        "avg_confidence": (
            (sum(r.confidence for r in results) / total) if total else 0.0
        ),
    }


# ------------------------------------------------------------------
# Save failures
# ------------------------------------------------------------------
def _save_failures(
    failures: list[PredictionResult],
    metrics: dict,
    *,
    split: DatasetSplit = "test",
):
    """Write detailed reasoning for every misclassified sample to a file."""
    failures_path = failures_path_for_split(split)
    lines: list[str] = []
    lines.append("# Failed Predictions\n")
    lines.append(
        f"**{len(failures)}** misclassified out of "
        f"**{metrics['total']}** samples "
        f"(accuracy {metrics['accuracy']:.2%})\n"
    )

    for idx, r in enumerate(failures, 1):
        lines.append("---\n")
        lines.append(f"## [{idx}] {r.headline}\n")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Ground truth | **{r.ground_truth.upper()}** |")
        lines.append(f"| Predicted | **{r.prediction.upper()}** |")
        lines.append(f"| Confidence | {r.confidence:.0%} |")
        lines.append("")

        # Matched community & explanation
        if r.matched_community:
            lines.append(f"### Matched community: **{r.matched_community}**\n")
            if r.community_explanation:
                lines.append(f"> {r.community_explanation}\n")

        # Document summary
        if r.doc_summary:
            lines.append("### Document summary\n")
            lines.append(f"> {r.doc_summary}\n")

        # Reasoning chain
        lines.append("### Reasoning chain\n")
        for part in r.reasoning.split(". "):
            part = part.strip()
            if part:
                lines.append(f"- {part}")
        lines.append("")

        # LLM explanation
        if r.llm_explanation:
            lines.append("### LLM explanation\n")
            lines.append(f"> {r.llm_explanation}\n")

        # Community similarity stats
        if r.community_stats:
            s = r.community_stats
            lines.append("### Community similarity\n")
            lines.append("| Metric | Fake | Real | Δ |")
            lines.append("|--------|------|------|---|")
            lines.append(
                f"| Max sim | {s.get('fake_max_sim', 0):.4f} "
                f"| {s.get('real_max_sim', 0):.4f} "
                f"| {s.get('diff_max_sim', 0):+.4f} |"
            )
            lines.append(
                f"| Mean sim | {s.get('fake_mean_sim', 0):.4f} "
                f"| {s.get('real_mean_sim', 0):.4f} "
                f"| {s.get('diff_mean_sim', 0):+.4f} |"
            )
            lines.append("")

        # Most similar communities
        tc = r.top_communities
        if tc.get("top_fake"):
            best = tc["top_fake"][0]
            lines.append("### Most similar FAKE community\n")
            lines.append(
                f"**{best['title']}** "
                f"(cosine similarity: {best['similarity']:.4f})\n"
            )
            lines.append(f"> {best['summary']}\n")
            if best.get("findings"):
                lines.append("Findings:")
                for f in best["findings"][:3]:
                    lines.append(f"- {f}")
                lines.append("")

        if tc.get("top_real"):
            best = tc["top_real"][0]
            lines.append("### Most similar REAL community\n")
            lines.append(
                f"**{best['title']}** "
                f"(cosine similarity: {best['similarity']:.4f})\n"
            )
            lines.append(f"> {best['summary']}\n")
            if best.get("findings"):
                lines.append("Findings:")
                for f in best["findings"][:3]:
                    lines.append(f"- {f}")
                lines.append("")

    text = "\n".join(lines)
    failures_path.write_text(text, encoding="utf-8")
    print(f"  → Saved to {failures_path}")


def _save_all_results(
    results: list[PredictionResult],
    metrics: dict,
    *,
    split: DatasetSplit = "test",
):
    """Write every prediction to a CSV file."""
    import csv

    all_results_path = results_path_for_split(split)

    max_k = max(
        (
            max(len(r.top_communities.get("top_fake", [])),
                len(r.top_communities.get("top_real", [])))
            for r in results if r.top_communities
        ),
        default=0,
    )

    header = [
        "id", "split", "title", "predicted", "actual", "correct", "confidence",
        "lgbm_pred", "lgbm_conf", "llm_decision", "llm_conf",
        "matched_community", "explanation",
        "fake_max_sim", "real_max_sim", "delta_max_sim",
    ]
    for i in range(1, max_k + 1):
        header.append(f"fake_community_{i}")
        header.append(f"fake_community_{i}_sim")
    for i in range(1, max_k + 1):
        header.append(f"real_community_{i}")
        header.append(f"real_community_{i}_sim")

    with open(all_results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for r in results:
            expl = r.community_explanation or r.llm_explanation or ""
            s = r.community_stats
            tc = r.top_communities

            row = [
                r.sample_id,
                r.split or split,
                r.headline,
                r.prediction,
                r.ground_truth,
                r.is_correct,
                f"{r.confidence:.4f}",
                r.lgbm_pred,
                f"{r.lgbm_conf:.4f}",
                r.llm_decision,
                f"{r.llm_conf:.4f}",
                r.matched_community or "",
                expl,
                f"{s.get('fake_max_sim', 0):.4f}" if s else "",
                f"{s.get('real_max_sim', 0):.4f}" if s else "",
                f"{s.get('diff_max_sim', 0):.4f}" if s else "",
            ]

            for c in tc.get("top_fake", [])[:max_k]:
                row.append(c["title"])
                row.append(f"{c['similarity']:.4f}")
            for _ in range(max_k - len(tc.get("top_fake", []))):
                row.extend(["", ""])

            for c in tc.get("top_real", [])[:max_k]:
                row.append(c["title"])
                row.append(f"{c['similarity']:.4f}")
            for _ in range(max_k - len(tc.get("top_real", []))):
                row.extend(["", ""])

            writer.writerow(row)

    print(f"  → All results saved to {all_results_path}")


def print_results(
    metrics: dict,
    results: list[PredictionResult],
    *,
    split: DatasetSplit = "test",
):
    print("\n" + "=" * 60)
    print(f"{split.upper()} RESULTS")
    print("=" * 60)
    print(f"\n  Total samples : {metrics['total']}")
    print(f"  Correct       : {metrics['correct']}")
    print(f"  Accuracy      : {metrics['accuracy']:.2%}")
    print(f"  Precision     : {metrics['precision']:.2%}")
    print(f"  Recall        : {metrics['recall']:.2%}")
    print(f"  F1            : {metrics['f1']:.2%}")
    print(f"  Avg confidence: {metrics['avg_confidence']:.2%}")

    _save_all_results(results, metrics, split=split)

    failures = [r for r in results if not r.is_correct]
    if failures:
        print(f"\n  Misclassified: {len(failures)} / {metrics['total']}")
        _save_failures(failures, metrics, split=split)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
async def main(
    limit: int | None = None,
    verbose: bool = True,
    balanced: bool = True,
    split: DatasetSplit = "test",
    llm_url: str | None = DEFAULT_LLM_URL,
    llm_model: str | None = None,
    embed_url: str = DEFAULT_EMBED_URL,
    embed_model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    no_communities: bool = False,
    no_llm: bool = False,
    use_cot: bool = True,
    fake_communities: str | None = None,
    real_communities: str | None = None,
    linear_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    linear_threshold: float = 0.5,
):
    print("=" * 60)
    print("Fake News Prediction")
    print("=" * 60)

    # 1. LightGBM
    lgbm = LightGBMClassifier()
    if not lgbm.load():
        raise RuntimeError(
            "Model not found. Train first: python scripts/train_classifier.py --tune"
        )

    # 2. Community similarity (optional)
    fake_path = Path(fake_communities) if fake_communities else None
    real_path = Path(real_communities) if real_communities else None

    community_sim = None
    if not no_communities:
        try:
            print("\n  Importing CommunitySimilarity…", flush=True)
            from narrative_map.prediction.community_similarity import CommunitySimilarity

            print("  Loading community embeddings…", flush=True)
            community_sim = CommunitySimilarity(
                embed_base_url=embed_url,
                embed_model=embed_model,
                fake_path=fake_path,
                real_path=real_path,
            )
        except Exception as exc:
            print(f"  ⚠ Community similarity unavailable: {exc}", flush=True)

    # 3. DSPy judge
    dspy_judge_instance = None
    resolved_model = llm_model

    if not no_llm:
        try:
            print("\n  Importing DSPy judge…", flush=True)
            from narrative_map.prediction.dspy_judge import configure_dspy_lm, load_compiled
            print("  Configuring DSPy LM…", flush=True)
            resolved_model = configure_dspy_lm(
                base_url=llm_url, model=llm_model,
            )
            print(f"  ✓ DSPy LM: {resolved_model} @ {llm_url}", flush=True)

            dspy_judge_instance = load_compiled(
                fake_path=fake_path, real_path=real_path,
            )
            if dspy_judge_instance is None:
                print("  No compiled module found; using base DSPy judge", flush=True)
                print(
                    "  Tip: run `python -m narrative_map.prediction.dspy_judge --compile` "
                    "to optimize with few-shot examples", flush=True,
                )
                from narrative_map.prediction.dspy_judge import DSPyJudge
                dspy_judge_instance = DSPyJudge.from_communities(
                    fake_path=fake_path, real_path=real_path,
                    use_cot=use_cot,
                )
            print(f"  ✓ DSPy judge ready (CoT={use_cot})", flush=True)
        except Exception as exc:
            print(f"  ⚠ DSPy judge unavailable ({exc}); falling back to LightGBM", flush=True)

    # 4. Load evaluation data
    eval_data = load_dataset_split(split, limit=limit, balanced=balanced)

    # 5. Pre-compute DSPy article summaries for community embedding (cache on disk)
    summary_cache = None
    summary_model: str | None = resolved_model
    if community_sim is not None and not summary_model and llm_url:
        from narrative_map.prediction.dspy_judge import configure_dspy_lm

        print(
            "\n  Resolving LLM for DSPy retrieval summaries…", flush=True,
        )
        summary_model = configure_dspy_lm(
            base_url=llm_url, model=llm_model,
        )
    if community_sim is not None and llm_url and summary_model:
        print("\nPre-computing article summaries (DSPy)…")
        summary_cache = SummaryCache()
        summary_cache = await precompute_summaries(
            eval_data, llm_url, summary_model, summary_cache,
        )
    elif community_sim is not None and not summary_model:
        print(
            "\n  ⚠ No LLM for DSPy summaries: community sim will fall back to "
            "headline + truncated body (set --llm-url or run with LLM).",
            flush=True,
        )

    # 6. Build predictor
    predictor = GraphRAGPredictor(
        lgbm_classifier=lgbm,
        community_sim=community_sim,
        top_k=top_k,
        dspy_judge=dspy_judge_instance,
        linear_weights=linear_weights,
        linear_threshold=linear_threshold,
        summary_cache=summary_cache,
        summary_llm_url=llm_url if community_sim is not None else None,
        summary_model=summary_model,
    )
    # 7. Run
    mode = (
        f"LightGBM + Community Sim + DSPy (CoT={use_cot})"
        if dspy_judge_instance
        else "LightGBM + Community Sim"
        if community_sim
        else "LightGBM only"
    )
    print(f"\nRunning predictions on {len(eval_data)} samples ({mode})…")
    results = await run_predictions(
        predictor, eval_data,
        summary_cache=summary_cache,
        verbose=verbose,
        split=split,
    )
    metrics = compute_metrics(results)
    print_results(metrics, results, split=split)
    return results, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM + Community Sim + LLM Judge"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        help="Use full unbalanced split (default in predict when set)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "validation", "test"),
        default="test",
        help="HF dataset split to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=DEFAULT_LLM_URL,
        help="OpenAI-compatible LLM base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model name (auto-detected by default)",
    )
    parser.add_argument(
        "--embed-url",
        type=str,
        default=DEFAULT_EMBED_URL,
        help="OpenAI-compatible embedding API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="Override embedding model name (auto-detected by default)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k communities per set (default: %(default)s)",
    )
    parser.add_argument(
        "--no-communities",
        action="store_true",
        help="Disable community similarity (LightGBM only)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM/DSPy judge (LightGBM + community stats only)",
    )
    parser.add_argument(
        "--no-cot",
        action="store_false",
        dest="use_cot",
        help="Disable Chain of Thought in DSPy judge (use standard Predict)",
    )
    parser.set_defaults(use_cot=True)
    parser.add_argument(
        "--fake-communities",
        type=str,
        default=None,
        help="Path to fake community reports (JSON or parquet). "
             "Default: communities/fake.json or .parquet",
    )
    parser.add_argument(
        "--real-communities",
        type=str,
        default=None,
        help="Path to real community reports (JSON or parquet). "
             "Default: communities/real.json or .parquet",
    )
    parser.add_argument(
        "--linear-threshold",
        type=float,
        default=0.5,
        help="P(real) decision threshold for linear_v1 (default: %(default)s)",
    )
    parser.add_argument(
        "--linear-w",
        nargs=3,
        type=float,
        metavar=("W_LGBM", "W_LLM", "W_SIM"),
        default=None,
        help="Weights for linear_v1: LGBM P(real), LLM P(real), community tilt",
    )
    args = parser.parse_args()

    lin_w: tuple[float, float, float]
    if args.linear_w is not None:
        lin_w = (args.linear_w[0], args.linear_w[1], args.linear_w[2])
    else:
        lin_w = (1 / 3, 1 / 3, 1 / 3)

    asyncio.run(
        main(
            limit=args.limit,
            verbose=not args.quiet,
            balanced=not args.unbalanced,
            split=args.split,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            top_k=args.top_k,
            no_communities=args.no_communities,
            no_llm=args.no_llm,
            use_cot=args.use_cot,
            fake_communities=args.fake_communities,
            real_communities=args.real_communities,
            linear_weights=lin_w,
            linear_threshold=args.linear_threshold,
        )
    )
