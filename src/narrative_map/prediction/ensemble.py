"""
Ensemble: weighted linear mix of LightGBM, LLM/judge, and community similarity (linear_v1).

Final score is a single ``P(real)`` from LGBM P(real), LLM P(real), and a community tilt
in [0, 1], then one threshold on ``P(real)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DecisionContext:
    """Normalized inputs to final prediction decision logic."""

    lgbm_pred: str
    lgbm_conf: float
    llm_pred: str = ""
    llm_conf: float = 0.0
    sim_stats: dict[str, float] = field(default_factory=dict)
    source: str = "none"
    has_judge: bool = False


@dataclass(frozen=True)
class DecisionOutcome:
    """Structured policy output used for auditability and testing."""

    label: str
    confidence: float
    did_override: bool
    winner: str
    reason_code: str
    debug: dict[str, Any] = field(default_factory=dict)


def _prob_real(pred_label: str, conf: float) -> float:
    pred_label = (pred_label or "").strip().lower()
    c = float(np.clip(conf, 0.0, 1.0))
    if pred_label == "real":
        return c
    if pred_label == "fake":
        return 1.0 - c
    return 0.5


def normalize_label(label: str, fallback: str = "fake") -> str:
    v = (label or "").strip().lower()
    if v in ("fake", "real"):
        return v
    fb = (fallback or "").strip().lower()
    if fallback == "":
        return ""
    if fb in ("fake", "real"):
        return fb
    return "fake"


def normalize_conf(conf: float) -> float:
    return float(np.clip(float(conf), 0.0, 1.0))


def make_decision_context(
    *,
    lgbm_pred: str,
    lgbm_conf: float,
    llm_pred: str = "",
    llm_conf: float = 0.0,
    sim_stats: dict | None = None,
    source: str = "none",
    has_judge: bool | None = None,
) -> DecisionContext:
    base_label = normalize_label(lgbm_pred, fallback="fake")
    judge_label = normalize_label(llm_pred, fallback="")
    judge_enabled = bool(has_judge) if has_judge is not None else bool(judge_label)
    sim = sim_stats or {}
    return DecisionContext(
        lgbm_pred=base_label,
        lgbm_conf=normalize_conf(lgbm_conf),
        llm_pred=judge_label,
        llm_conf=normalize_conf(llm_conf),
        sim_stats={
            "fake_max_sim": float(sim.get("fake_max_sim", 0) or 0),
            "real_max_sim": float(sim.get("real_max_sim", 0) or 0),
        },
        source=source or "none",
        has_judge=judge_enabled,
    )


def ensemble_linear_v1(
    lgbm_pred: str,
    lgbm_conf: float,
    llm_pred: str,
    llm_conf: float,
    sim_stats: dict,
    *,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    decision_threshold: float = 0.5,
) -> tuple[str, float, bool, float]:
    """
    Single scalar ``P(real)`` from LGBM P(real), LLM P(real), and community tilt.

    Community tilt maps ``real_max - fake_max`` to [0, 1] via ``(delta + 1) / 2``.

    Returns ``(pred, confidence, did_override, p_real)``.
    """
    w = np.array(weights, dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    else:
        w = w / s

    p_lgbm = _prob_real(lgbm_pred, lgbm_conf)
    ll = (llm_pred or "").strip().lower()
    if ll not in ("fake", "real"):
        p_llm = p_lgbm
    else:
        p_llm = _prob_real(llm_pred, llm_conf)

    if sim_stats:
        fake_max = float(sim_stats.get("fake_max_sim", 0) or 0)
        real_max = float(sim_stats.get("real_max_sim", 0) or 0)
        delta = np.clip(real_max - fake_max, -1.0, 1.0)
        p_sim = float((delta + 1.0) / 2.0)
    else:
        p_sim = 0.5

    p_real = float(w[0] * p_lgbm + w[1] * p_llm + w[2] * p_sim)
    pred = "real" if p_real >= decision_threshold else "fake"
    conf = p_real if pred == "real" else (1.0 - p_real)

    base = (lgbm_pred or "").strip().lower()
    if base not in ("fake", "real"):
        base = "fake"
    did_override = pred != base
    return pred, conf, did_override, p_real


def _build_outcome(
    context: DecisionContext,
    pred: str,
    conf: float,
    did_override: bool,
    *,
    reason_code: str,
    p_real: float | None = None,
) -> DecisionOutcome:
    label = normalize_label(pred, fallback=context.lgbm_pred)
    confidence = normalize_conf(conf)
    if label == context.lgbm_pred and label != context.llm_pred:
        winner = "lgbm"
    elif label == context.llm_pred and label != context.lgbm_pred:
        winner = "llm"
    elif label == context.lgbm_pred == context.llm_pred:
        winner = "agreement"
    else:
        winner = "ensemble"
    dbg = {
        "source": context.source,
        "has_judge": context.has_judge,
        "lgbm_pred": context.lgbm_pred,
        "lgbm_conf": context.lgbm_conf,
        "llm_pred": context.llm_pred,
        "llm_conf": context.llm_conf,
        "sim_stats": context.sim_stats,
    }
    if p_real is not None:
        dbg["p_real"] = float(p_real)
    return DecisionOutcome(
        label=label,
        confidence=confidence,
        did_override=bool(did_override),
        winner=winner,
        reason_code=reason_code,
        debug=dbg,
    )


def ensemble_decide(
    context: DecisionContext,
    *,
    linear_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    linear_threshold: float = 0.5,
) -> DecisionOutcome:
    """Apply linear_v1 and return a structured decision outcome."""
    pred, conf, ov, p_r = ensemble_linear_v1(
        context.lgbm_pred,
        context.lgbm_conf,
        context.llm_pred,
        context.llm_conf,
        context.sim_stats,
        weights=linear_weights,
        decision_threshold=linear_threshold,
    )
    return _build_outcome(context, pred, conf, ov, reason_code="linear_v1", p_real=p_r)


def ensemble_forward(
    lgbm_pred: str,
    lgbm_conf: float,
    llm_pred: str,
    llm_conf: float,
    sim_stats: dict,
    *,
    linear_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    linear_threshold: float = 0.5,
) -> tuple[str, float, bool]:
    """Run linear_v1 ensemble strategy."""
    context = make_decision_context(
        lgbm_pred=lgbm_pred,
        lgbm_conf=lgbm_conf,
        llm_pred=llm_pred,
        llm_conf=llm_conf,
        sim_stats=sim_stats,
        source="ensemble_forward",
        has_judge=True,
    )
    outcome = ensemble_decide(
        context,
        linear_weights=linear_weights,
        linear_threshold=linear_threshold,
    )
    return outcome.label, outcome.confidence, outcome.did_override


def prob_real(pred_label: str, conf: float) -> float:
    """Map (label, confidence) to P(real)."""
    return _prob_real(pred_label, conf)


META_FEATURE_NAMES = [
    "p_lgbm_real",
    "p_llm_real",
    "p_sim_real",
    "fake_max_sim",
    "real_max_sim",
    "delta_max_sim",
]


def build_meta_features(
    lgbm_pred: str,
    lgbm_conf: float,
    llm_pred: str,
    llm_conf: float,
    sim_stats: dict,
) -> np.ndarray:
    """Feature vector for stack_lr meta-classifier."""
    p_lgbm = prob_real(lgbm_pred, lgbm_conf)
    ll = (llm_pred or "").strip().lower()
    p_llm = prob_real(llm_pred, llm_conf) if ll in ("fake", "real") else p_lgbm

    fake_max = float(sim_stats.get("fake_max_sim", 0) or 0)
    real_max = float(sim_stats.get("real_max_sim", 0) or 0)
    delta = float(np.clip(real_max - fake_max, -1.0, 1.0))
    p_sim = float((delta + 1.0) / 2.0)

    return np.array(
        [p_lgbm, p_llm, p_sim, fake_max, real_max, delta],
        dtype=np.float64,
    )


def ensemble_predict_pair(
    lgbm_pred: str,
    lgbm_conf: float,
    llm_pred: str,
    llm_conf: float,
    sim_stats: dict,
    **kwargs: Any,
) -> tuple[str, float]:
    """Like ``ensemble_forward`` but drops the override flag (legacy API)."""
    pred, conf, _ = ensemble_forward(
        lgbm_pred, lgbm_conf, llm_pred, llm_conf, sim_stats, **kwargs
    )
    return pred, conf
