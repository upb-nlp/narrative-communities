"""
DSPy-based fake news classifier for Romanian news articles.

Replaces the raw LLM judge (which fails to produce structured output with
DeepSeek thinking models) with DSPy modules that:
1. Enforce structured output parsing with automatic retries
2. Support few-shot bootstrapping from training data
3. Can be compiled/optimized with BootstrapFewShot or MIPROv2

Usage:
    # Standalone classification
    judge = DSPyJudge.from_communities("communities/")
    result = await judge.judge(headline, body, top_fake=..., top_real=..., stats=...)
    # result is a JudgeResult(decision, confidence, reasoning, matched_community, explanation)

    # Compile on training data (run once, saves to disk)
    python -m narrative_map.prediction.dspy_judge --compile
    # Match predict-time top-k community strings (needs embedding server):
    python -m narrative_map.prediction.dspy_judge --compile --with-retrieval
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from narrative_map.config import COMMUNITIES_DIR, DEFAULT_LLM_URL, DSPY_COMPILED_PATH

logger = logging.getLogger(__name__)

COMPILED_PATH = DSPY_COMPILED_PATH

# Tuned for 16K context window: patterns + article + few-shot demos must fit
MAX_PATTERNS_PER_CORPUS = 6
MAX_PATTERN_CHARS = 120
MAX_ARTICLE_CHARS = 1200
DEFAULT_TOP_K = 5
MAX_COMMUNITY_CHARS = 1000


def format_top_communities_for_judge(
    top_fake: list[dict] | None,
    top_real: list[dict] | None,
    *,
    max_per_side: int = 2,
    max_chars: int = MAX_COMMUNITY_CHARS,
) -> tuple[str, str]:
    """Format retrieved communities the same way as :meth:`DSPyJudge.judge`."""

    def fmt(comms: list[dict] | None) -> str:
        if not comms:
            return "(indisponibil)"
        lines = []
        for c in comms[:max_per_side]:
            title = c.get("title", "Comunitate")
            summ = str(c.get("summary", ""))[:max_chars]
            sim = float(c.get("similarity", 0))
            lines.append(f"• {title} (Sim: {sim:.2f}): {summ}")
        return "\n".join(lines)

    return fmt(top_fake), fmt(top_real)


@dataclass
class JudgeResult:
    """Result from the DSPy judge with community-grounded explanation."""

    decision: str
    confidence: float
    reasoning: str
    matched_community: str
    explanation: str


# ---------------------------------------------------------------------------
# Lazy DSPy class initialization — avoids importing dspy at module load time
# (dspy pulls in litellm + dozens of providers, which can deadlock on macOS)
# ---------------------------------------------------------------------------

_dspy_classes_cache: dict = {}


def _get_dspy():
    """Import dspy lazily and return it.

    Disables litellm's network calls at import time (telemetry, model cost
    map fetching) which can hang on firewalled / HPC networks.
    """
    import os
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_TELEMETRY", "False")
    os.environ.setdefault("LITELLM_LOG", "ERROR")

    import dspy
    return dspy


def _get_dspy_classes():
    """Create and cache the DSPy Signature and Module subclasses."""
    if _dspy_classes_cache:
        return _dspy_classes_cache

    dspy = _get_dspy()

    class FakeNewsClassification(dspy.Signature):
        """Clasifică un articol românesc ca FAKE sau REAL. Jurnalismul despre subiecte controversate este REAL. FAKE = senzaționalism, tratamente nevalidate, narativ anti-sistem, conspirații.
        IMPORTANT: Dacă articolul se potrivește cu temele și tacticile dintr-o comunitate FAKE de mai jos, probabil este FAKE, chiar dacă pare scris profesionist."""

        article = dspy.InputField(desc="Articol de știri românesc")
        fake_patterns = dspy.InputField(desc="Pattern-uri FAKE din graf de cunoștințe")
        real_patterns = dspy.InputField(desc="Pattern-uri REAL din graf de cunoștințe")
        fake_community_summaries = dspy.InputField(desc="Rezumatul celor mai similare comunități FAKE din graf (conțin tacticile lor)")
        real_community_summaries = dspy.InputField(desc="Rezumatul celor mai similare comunități REAL din graf (conțin stilul lor)")

        reasoning = dspy.OutputField(desc="2-3 propoziții: cum se compară tacticile articolului cu rezumatele comunităților fake vs real")
        classification = dspy.OutputField(desc="FAKE sau REAL")
        confidence = dspy.OutputField(desc="Număr între 0.50 și 1.00")
        matched_community = dspy.OutputField(
            desc="Numele exact al comunității din datele de mai sus care se potrivește cel mai bine cu articolul"
        )
        explanation = dspy.OutputField(
            desc="2-3 propoziții: de ce articolul aparține acestei comunități, ce pattern-uri comune au, și de ce este FAKE sau REAL"
        )

    class ArticleSummarization(dspy.Signature):
        """Rezumat neutru al articolului pentru similaritate semantică cu rezumatele comunităților din graf (aceeași granularitate ca un „community summary”). NU clasifica FAKE/REAL. Concentrează-te pe teme, entități, narațiune și fapte raportate."""

        headline = dspy.InputField(desc="Titlul articolului")
        body = dspy.InputField(desc="Corpul articolului")
        summary = dspy.OutputField(
            desc="Rezumat scurt (max ~300 cuvinte), persoana a 3-a, fără verdict FAKE/REAL, "
            "comparabil ca stil cu un rezumat de comunitate (teme, actori, fapte principale)."
        )

    class DSPySummarizer(dspy.Module):
        """Article summarizer using DSPy."""

        def __init__(self, use_cot: bool = False):
            super().__init__()
            cls = _get_dspy_classes()
            if use_cot:
                self.summarize = dspy.ChainOfThought(cls["ArticleSummarization"])
            else:
                self.summarize = dspy.Predict(cls["ArticleSummarization"])

        def forward(self, headline, body):
            return self.summarize(headline=headline, body=body[:3000])

    class DSPyJudge(dspy.Module):
        """Fake news classifier using ChainOfThought or Predict with community patterns."""

        def __init__(self, fake_patterns: str = "", real_patterns: str = "", use_cot: bool = True):
            super().__init__()
            cls = _get_dspy_classes()
            if use_cot:
                self.classify = dspy.ChainOfThought(cls["FakeNewsClassification"])
            else:
                self.classify = dspy.Predict(cls["FakeNewsClassification"])
            self._fake_patterns = fake_patterns
            self._real_patterns = real_patterns
            self.use_cot = use_cot

        @classmethod
        def from_communities(
            cls_ref,
            communities_dir: Path = COMMUNITIES_DIR,
            max_patterns: int = MAX_PATTERNS_PER_CORPUS,
            fake_path: Path | None = None,
            real_path: Path | None = None,
            use_cot: bool = True,
        ) -> "DSPyJudge":
            """Load community findings and build the judge."""
            from narrative_map.prediction.community_similarity import _resolve_community_file

            fake_file = fake_path or _resolve_community_file(communities_dir, "fake")
            real_file = real_path or _resolve_community_file(communities_dir, "real")

            fake_findings = load_findings(fake_file)
            real_findings = load_findings(real_file)
            logger.info(
                f"Loaded {len(fake_findings)} fake + {len(real_findings)} real findings"
            )

            cls = _get_dspy_classes()
            return cls["DSPyJudge"](
                fake_patterns=aggregate_patterns(fake_findings, max_patterns),
                real_patterns=aggregate_patterns(real_findings, max_patterns),
                use_cot=use_cot,
            )

        def forward(self, article, fake_patterns=None, real_patterns=None,
                    fake_community_summaries=None, real_community_summaries=None):
            return self.classify(
                article=article[:MAX_ARTICLE_CHARS],
                fake_patterns=fake_patterns or self._fake_patterns,
                real_patterns=real_patterns or self._real_patterns,
                fake_community_summaries=fake_community_summaries or "(indisponibil)",
                real_community_summaries=real_community_summaries or "(indisponibil)",
            )

        async def judge(
            self,
            headline: str,
            body: str,
            top_fake: list[dict] | None = None,
            top_real: list[dict] | None = None,
            stats: dict | None = None,
            top_k: int = DEFAULT_TOP_K,
        ) -> JudgeResult:
            """Classify an article with community-grounded explanation using Narrative RAG."""
            article_text = f"Titlu: {headline}\n\nText: {body[:MAX_ARTICLE_CHARS]}"

            f_comm, r_comm = format_top_communities_for_judge(top_fake, top_real)

            loop = asyncio.get_event_loop()
            result = None

            for attempt in range(2):
                try:
                    # Pass summaries on first attempt, fallback to "(indisponibil)" if too long
                    fc = f_comm if attempt == 0 else "(indisponibil)"
                    rc = r_comm if attempt == 0 else "(indisponibil)"
                    
                    result = await loop.run_in_executor(
                        None,
                        lambda: self(
                            article=article_text,
                            fake_community_summaries=fc,
                            real_community_summaries=rc,
                        ),
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    if attempt == 0 and ("ContextWindow" in err_str or "context length" in err_str.lower()):
                        logger.warning("Context too long, retrying without community summaries")
                        continue
                    if attempt == 0 and ("JSON" in err_str or "serialize" in err_str):
                        logger.warning("JSON parse failed, retrying without community summaries")
                        continue
                    logger.error(f"DSPy classification failed: {e}")
                    return JudgeResult(
                        decision="fake", confidence=0.50,
                        reasoning=f"Eroare DSPy: {err_str[:200]}",
                        matched_community="", explanation="",
                    )

            if result is None:
                return JudgeResult(
                    decision="fake", confidence=0.50,
                    reasoning="DSPy: nu s-a putut clasifica",
                    matched_community="", explanation="",
                )

            decision = _parse_classification(
                result.classification if hasattr(result, "classification") else ""
            )
            confidence = _parse_confidence(
                result.confidence if hasattr(result, "confidence") else ""
            )
            reasoning = (
                result.reasoning.strip()
                if hasattr(result, "reasoning") and result.reasoning
                else ""
            )
            matched_community = (
                result.matched_community.strip()
                if hasattr(result, "matched_community") and result.matched_community
                else ""
            )
            explanation = (
                result.explanation.strip()
                if hasattr(result, "explanation") and result.explanation
                else ""
            )

            if decision is None:
                decision = _parse_classification(reasoning)
            if decision is None:
                decision = "fake"
                confidence = 0.50

            return JudgeResult(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                matched_community=matched_community,
                explanation=explanation,
            )

    _dspy_classes_cache["FakeNewsClassification"] = FakeNewsClassification
    _dspy_classes_cache["ArticleSummarization"] = ArticleSummarization
    _dspy_classes_cache["DSPySummarizer"] = DSPySummarizer
    _dspy_classes_cache["DSPyJudge"] = DSPyJudge
    return _dspy_classes_cache


# Module-level lazy access: ``from narrative_map.prediction.dspy_judge import DSPyJudge``
def __getattr__(name: str):
    if name in ("DSPyJudge", "FakeNewsClassification", "DSPySummarizer", "ArticleSummarization"):
        return _get_dspy_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Community pattern loading & aggregation (no dspy dependency)
# ---------------------------------------------------------------------------

def load_findings(path: Path) -> list[dict]:
    """Extract findings from a community reports file (JSON or parquet)."""
    from narrative_map.prediction.community_similarity import load_communities

    try:
        df = load_communities(path)
    except Exception as e:
        logger.warning(f"Cannot read {path}: {e}")
        return []

    all_findings: list[dict] = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        findings_raw = row.get("findings", [])
        if isinstance(findings_raw, str):
            try:
                findings_raw = json.loads(findings_raw)
            except json.JSONDecodeError:
                findings_raw = []
        if findings_raw is None:
            continue
        try:
            if len(findings_raw) == 0:
                continue
        except TypeError:
            continue
        for f in findings_raw:
            if isinstance(f, dict):
                summary = f.get("summary", "").strip()
                explanation = f.get("explanation", "").strip()
                if summary or explanation:
                    all_findings.append({
                        "community": title,
                        "summary": summary,
                        "explanation": explanation,
                    })
    return all_findings


def _select_diverse(findings: list[dict], max_n: int) -> list[dict]:
    """Round-robin across communities for diversity."""
    if len(findings) <= max_n:
        return findings

    by_community: dict[str, list[dict]] = {}
    for f in findings:
        by_community.setdefault(f["community"], []).append(f)
    for key in by_community:
        by_community[key].sort(
            key=lambda x: len(x.get("explanation", "")), reverse=True
        )

    selected: list[dict] = []
    keys = list(by_community.keys())
    ptr = {k: 0 for k in keys}
    while len(selected) < max_n:
        added = False
        for k in keys:
            if ptr[k] < len(by_community[k]):
                selected.append(by_community[k][ptr[k]])
                ptr[k] += 1
                added = True
                if len(selected) >= max_n:
                    break
        if not added:
            break
    return selected


def aggregate_patterns(
    findings: list[dict],
    max_patterns: int = MAX_PATTERNS_PER_CORPUS,
    max_chars: int = MAX_PATTERN_CHARS,
) -> str:
    """Build a concise prompt-ready pattern string."""
    if not findings:
        return "(nu sunt disponibile pattern-uri)"

    selected = _select_diverse(findings, max_patterns)
    lines: list[str] = []
    current_community = None
    for f in selected:
        if f["community"] != current_community:
            current_community = f["community"]
            lines.append(f"\n[{current_community}]")
        text = f.get("explanation") or f.get("summary", "")
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        summary = f.get("summary", "")
        if summary and summary != text[:len(summary)]:
            lines.append(f"• {summary}: {text}")
        else:
            lines.append(f"• {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_classification(raw: str) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip().upper()
    if cleaned in ("FAKE", "REAL"):
        return cleaned.lower()
    if "FAKE" in cleaned and "REAL" not in cleaned:
        return "fake"
    if "REAL" in cleaned and "FAKE" not in cleaned:
        return "real"
    fake_pos = cleaned.rfind("FAKE")
    real_pos = cleaned.rfind("REAL")
    if fake_pos >= 0 and real_pos >= 0:
        return "real" if real_pos > fake_pos else "fake"
    return None


def _parse_confidence(raw: str) -> float:
    if not raw:
        return 0.70
    try:
        clean = re.sub(r"[^0-9.]", "", raw.strip())
        val = float(clean)
        if val > 1.0:
            val /= 100.0
        return max(0.50, min(1.0, val))
    except (ValueError, IndexError):
        return 0.70


# ---------------------------------------------------------------------------
# Compilation / optimization
# ---------------------------------------------------------------------------

def _accuracy_metric(example, pred, trace=None) -> bool:
    """Return True if predicted classification matches ground truth."""
    gt = example.get("classification", "").strip().upper()
    if not gt:
        gt = example.get("expected_classification", "").strip().upper()
        
    predicted = _parse_classification(
        pred.classification if hasattr(pred, "classification") else ""
    )
    if predicted is None:
        # Fallback to parsing reasoning if classification field is empty
        predicted = _parse_classification(
            pred.reasoning if hasattr(pred, "reasoning") else ""
        )
    
    if predicted is None:
        return False
    return predicted == gt.lower()


def compile_classifier(
    llm_url: str = DEFAULT_LLM_URL,
    llm_model: str | None = None,
    communities_dir: Path = COMMUNITIES_DIR,
    fake_path: Path | None = None,
    real_path: Path | None = None,
    n_train: int = 40,
    n_bootstrap: int = 8,
    max_rounds: int = 2,
    save_path: Path = COMPILED_PATH,
    use_cot: bool = True,
    *,
    with_retrieval: bool = False,
    embed_url: str | None = None,
    embed_model: str | None = None,
    retrieval_top_k: int = DEFAULT_TOP_K,
):
    """Compile the DSPy judge on training data using BootstrapFewShot.

    Set ``with_retrieval=True`` to fill ``fake_community_summaries`` /
    ``real_community_summaries`` from ``CommunitySimilarity`` (same retrieval
    path as inference). Requires the embedding API reachable at ``embed_url``.
    """
    from datasets import load_dataset
    dspy = _get_dspy()
    cls = _get_dspy_classes()

    if llm_model is None:
        from openai import OpenAI
        client = OpenAI(base_url=llm_url, api_key="no-key")
        models = client.models.list()
        llm_model = models.data[0].id if models.data else "default"

    lm = dspy.LM(
        model=f"openai/{llm_model}",
        api_base=llm_url,
        api_key="no-key",
        max_tokens=1000,
        temperature=0.1,
    )
    dspy.configure(lm=lm)

    judge = cls["DSPyJudge"].from_communities(
        communities_dir,
        fake_path=fake_path,
        real_path=real_path,
        use_cot=use_cot,
    )

    print(f"Loading training data from upb-nlp/ro_fake_news…")
    ds = load_dataset("upb-nlp/ro_fake_news", "default")
    train_data = ds["train"].to_list() if "train" in ds else []
    if "validation" in ds:
        train_data.extend(ds["validation"].to_list())

    import random
    random.seed(42)

    real_news_label = "Știre reală"
    real_samples = [s for s in train_data if s.get("supernarrative") == real_news_label]
    fake_samples = [s for s in train_data if s.get("supernarrative") != real_news_label]

    random.shuffle(real_samples)
    random.shuffle(fake_samples)

    n_per_class = n_train // 2
    selected = real_samples[:n_per_class] + fake_samples[:n_per_class]
    random.shuffle(selected)

    community_sim = None
    if with_retrieval:
        from narrative_map.prediction.community_similarity import (
            CommunitySimilarity,
            DEFAULT_EMBED_URL,
        )

        embed_base = embed_url or DEFAULT_EMBED_URL
        print(
            f"Embedding communities + per-article retrieval ({embed_base}, "
            f"top_k={retrieval_top_k})…"
        )
        community_sim = CommunitySimilarity(
            communities_dir=communities_dir,
            embed_base_url=embed_base,
            embed_model=embed_model,
            fake_path=fake_path,
            real_path=real_path,
        )

    examples: list = []
    for i, sample in enumerate(selected):
        headline = sample.get("headline", "")
        body = sample.get("body", "")
        is_real = sample.get("supernarrative", "").strip() == real_news_label
        article_text = f"Titlu: {headline}\n\nText: {body[:MAX_ARTICLE_CHARS]}"

        if community_sim is not None:
            query_text = (
                f"{headline} {body[:2000]}" if body else headline
            )
            q = community_sim.query(query_text, top_k=retrieval_top_k)
            fc, rc = format_top_communities_for_judge(
                q["top_fake"], q["top_real"]
            )
            if (i + 1) % 10 == 0:
                print(f"  retrieval: {i + 1}/{len(selected)} examples…")
        else:
            fc, rc = "(indisponibil)", "(indisponibil)"

        ex = dspy.Example(
            article=article_text,
            fake_patterns=judge._fake_patterns,
            real_patterns=judge._real_patterns,
            fake_community_summaries=fc,
            real_community_summaries=rc,
            classification="REAL" if is_real else "FAKE",
        ).with_inputs("article", "fake_patterns", "real_patterns", "fake_community_summaries", "real_community_summaries")
        examples.append(ex)

    print(f"Prepared {len(examples)} training examples ({n_per_class} per class)")

    split = int(len(examples) * 0.7)
    train_set = examples[:split]
    eval_set = examples[split:]

    print(f"Training: {len(train_set)}, Eval: {len(eval_set)}")
    print(f"Compiling with BootstrapFewShot (max_rounds={max_rounds})…")

    optimizer = dspy.teleprompt.BootstrapFewShot(
        metric=_accuracy_metric,
        max_bootstrapped_demos=n_bootstrap,
        max_labeled_demos=2,
        max_rounds=max_rounds,
    )

    compiled_judge = optimizer.compile(
        judge,
        trainset=train_set,
    )

    compiled_judge.save(str(save_path))
    print(f"✓ Compiled module saved to {save_path}")

    correct = 0
    for ex in eval_set:
        try:
            pred = compiled_judge(
                article=ex.article,
                fake_community_summaries=ex.fake_community_summaries,
                real_community_summaries=ex.real_community_summaries,
            )
            if _accuracy_metric(ex, pred):
                correct += 1
        except Exception as e:
            print(f"  Eval error: {e}")
    print(f"  Eval accuracy: {correct}/{len(eval_set)} ({correct/max(len(eval_set),1):.1%})")

    return compiled_judge


def load_compiled(
    path: Path = COMPILED_PATH,
    communities_dir: Path = COMMUNITIES_DIR,
    fake_path: Path | None = None,
    real_path: Path | None = None,
):
    """Load a previously compiled DSPy judge from disk."""
    if not path.exists():
        logger.info(f"No compiled module at {path}; using uncompiled judge")
        return None
    try:
        cls = _get_dspy_classes()
        judge = cls["DSPyJudge"].from_communities(
            communities_dir,
            fake_path=fake_path,
            real_path=real_path,
        )
        judge.load(str(path))
        logger.info(f"Loaded compiled DSPy judge from {path}")
        return judge
    except Exception as e:
        logger.warning(f"Failed to load compiled module: {e}")
        return None


# ---------------------------------------------------------------------------
# Configure DSPy LM
# ---------------------------------------------------------------------------

def configure_dspy_lm(
    base_url: str = DEFAULT_LLM_URL,
    model: str | None = None,
):
    """Configure DSPy to use local LLM with appropriate settings."""
    dspy = _get_dspy()

    if model is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="no-key")
        models = client.models.list()
        model = models.data[0].id if models.data else "default"

    lm = dspy.LM(
        model=f"openai/{model}",
        api_base=base_url,
        api_key="no-key",
        max_tokens=1000,
        temperature=0.1,
    )
    dspy.configure(lm=lm)
    logger.info(f"DSPy configured: {model} @ {base_url}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DSPy fake news judge")
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile the DSPy module on training data",
    )
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-bootstrap", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--no-cot",
        action="store_false",
        dest="use_cot",
        help="Disable Chain of Thought during compilation",
    )
    parser.set_defaults(use_cot=True)
    parser.add_argument(
        "--communities-dir",
        type=Path,
        default=COMMUNITIES_DIR,
        help="Directory used to resolve default fake/real filenames (default: %(default)s)",
    )
    parser.add_argument(
        "--fake-communities",
        type=Path,
        default=None,
        help="Path to fake community reports (JSON or parquet). "
        "Default: <communities-dir>/fake.json or fake.parquet",
    )
    parser.add_argument(
        "--real-communities",
        type=Path,
        default=None,
        help="Path to real community reports (JSON or parquet). "
        "Default: <communities-dir>/real.json or real.parquet",
    )
    parser.add_argument(
        "--with-retrieval",
        action="store_true",
        help="For each training article, fill fake/real community summaries via "
        "the same embedding retrieval as predict.py (requires embedding server)",
    )
    parser.add_argument(
        "--embed-url",
        default=None,
        help="OpenAI-compatible embeddings API (default: narrative_map.community_similarity "
        "DEFAULT_EMBED_URL when --with-retrieval)",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Embedding model id (default: first model from /models)",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k communities per side for retrieval (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.compile:
        compile_classifier(
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            communities_dir=args.communities_dir,
            fake_path=args.fake_communities,
            real_path=args.real_communities,
            n_train=args.n_train,
            n_bootstrap=args.n_bootstrap,
            max_rounds=args.max_rounds,
            use_cot=args.use_cot,
            with_retrieval=args.with_retrieval,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            retrieval_top_k=args.retrieval_top_k,
        )
    else:
        parser.print_help()
