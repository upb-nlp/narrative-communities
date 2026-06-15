"""Threshold grid evaluation helpers (dev/test split + override-aware metrics)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from narrative_map.data import DatasetSplit, load_split_ids
from narrative_map.prediction.ensemble import ensemble_decide, make_decision_context
from narrative_map.prediction.predict import PredictionResult


def threshold_grid(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive-ish grid [lo, hi] with given step (floating-safe)."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round((hi - lo) / step)) + 1
    vals = [round(lo + i * step, 6) for i in range(n) if lo + i * step <= hi + 1e-9]
    if not vals or vals[-1] < hi - 1e-9:
        vals.append(round(hi, 6))
    return sorted(set(vals))


def filter_dataframe_by_split(df: pd.DataFrame, split: DatasetSplit) -> pd.DataFrame:
    """Keep rows belonging to an official HF split (by ``split`` or ``id`` column)."""
    if "split" in df.columns:
        mask = df["split"].astype(str).str.lower() == split
        if mask.any():
            return df.loc[mask].reset_index(drop=True)
    if "id" in df.columns:
        ids = load_split_ids(split)
        if ids:
            return df[df["id"].astype(str).isin(ids)].reset_index(drop=True)
    raise ValueError(
        f"Cannot filter CSV to split={split!r}: need a 'split' column or "
        f"article ids matching the official dataset."
    )


def filter_results_by_split(
    results: Sequence[PredictionResult],
    split: DatasetSplit,
) -> list[PredictionResult]:
    """Keep prediction rows for one official split."""
    if results and all(r.split for r in results):
        return [r for r in results if r.split == split]
    ids = load_split_ids(split)
    if ids:
        return [r for r in results if r.sample_id in ids]
    return list(results)


def normalized_llm_label(result: PredictionResult) -> str:
    """Match scripts/sweep_thresholds fallback when judge output is missing."""
    llm_label = (result.llm_decision or "").strip().lower()
    if llm_label not in ("fake", "real"):
        return result.lgbm_pred
    return llm_label


def _replay_ensemble_row(
    r: PredictionResult,
    *,
    linear_weights: tuple[float, float, float],
    linear_threshold: float,
) -> tuple[str, float, bool]:
    """
    Replay final label for one row using the same policy stack as ``GraphRAGPredictor``.

    Returns (pred_label, confidence, did_override).
    """
    llm_label = normalized_llm_label(r)
    raw = (r.llm_decision or "").strip().lower()
    has_judge = raw in ("fake", "real")
    ctx = make_decision_context(
        lgbm_pred=r.lgbm_pred,
        lgbm_conf=r.lgbm_conf,
        llm_pred=llm_label,
        llm_conf=r.llm_conf,
        sim_stats=r.community_stats or {},
        source="threshold_sweep",
        has_judge=has_judge,
    )
    out = ensemble_decide(
        ctx,
        linear_weights=linear_weights,
        linear_threshold=linear_threshold,
    )
    return out.label, out.confidence, out.did_override


def stratified_dev_test_indices(
    y_binary: Sequence[int],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split row indices into dev / held-out test.

    Stratifies by label when each class has at least 2 samples; otherwise
    falls back to a random split.
    """
    y = np.asarray(y_binary, dtype=int)
    n = len(y)
    idx = np.arange(n, dtype=int)
    if n == 0:
        return idx, idx
    if not (0.0 < dev_fraction < 1.0):
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    test_size = 1.0 - dev_fraction
    if n == 1:
        # Cannot stratify split a single row; treat all data as dev-only.
        return idx, np.array([], dtype=int)

    _, counts = np.unique(y, return_counts=True)
    stratify = y if len(counts) >= 2 and np.min(counts) >= 2 else None

    dev_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=stratify,
        random_state=seed,
        shuffle=True,
    )
    return np.sort(dev_idx), np.sort(test_idx)


def metrics_bundle(
    y_true: list[int],
    y_pred: list[int],
    overrides: list[bool],
) -> dict[str, float]:
    """
    y_true / y_pred are binary with positive class = real (1).

    override_rate is overrides_total / n_samples (not conditioned on disagreement).
    """
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "overrides_total": 0.0,
            "overrides_correct": 0.0,
            "override_precision": 0.0,
            "override_rate": 0.0,
        }

    overrides_total = int(sum(overrides))
    ov_correct = 0
    for yt, yp, ov in zip(y_true, y_pred, overrides):
        if ov and yp == yt:
            ov_correct += 1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "overrides_total": float(overrides_total),
        "overrides_correct": float(ov_correct),
        "override_precision": float(ov_correct / overrides_total)
        if overrides_total
        else 0.0,
        "override_rate": float(overrides_total / n),
    }


def evaluate_ensemble_on_indices(
    results: Sequence[PredictionResult],
    y_true: Sequence[int],
    indices: np.ndarray | Sequence[int],
    *,
    linear_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    linear_threshold: float = 0.5,
) -> dict[str, float]:
    """Replay linear_v1 ensemble for a subset of rows (by index)."""
    y_true_sub: list[int] = []
    y_pred: list[int] = []
    overrides: list[bool] = []

    for i in indices:
        r = results[i]
        yt = y_true[i]
        pred_label, _conf, did_override = _replay_ensemble_row(
            r,
            linear_weights=linear_weights,
            linear_threshold=linear_threshold,
        )
        y_true_sub.append(yt)
        y_pred.append(1 if pred_label == "real" else 0)
        overrides.append(did_override)

    return metrics_bundle(y_true_sub, y_pred, overrides)


def lgbm_baseline_on_indices(
    results: Sequence[PredictionResult],
    y_true: Sequence[int],
    indices: np.ndarray | Sequence[int],
) -> dict[str, float]:
    """LightGBM-only metrics on the same index subset."""
    y_true_sub: list[int] = []
    y_pred: list[int] = []
    for i in indices:
        r = results[i]
        y_true_sub.append(y_true[i])
        y_pred.append(1 if r.lgbm_pred == "real" else 0)
    fake_ov = [False] * len(y_true_sub)
    return metrics_bundle(y_true_sub, y_pred, fake_ov)


def ensemble_diagnostics_at_thresholds(
    results: Sequence[PredictionResult],
    *,
    linear_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    linear_threshold: float = 0.5,
) -> dict[str, int]:
    """
    Count agree / override / blocked using the same replay logic as the sweep.

    - Agree: normalized judge label equals LightGBM label.
    - Override accepted: disagree and ensemble takes the judge.
    - Override blocked: disagree and ensemble keeps LightGBM.
    """
    agree = 0
    override_accepted = 0
    override_blocked = 0
    overrides_correct = 0
    blocked_lgbm_wrong = 0

    for r in results:
        llm_label = normalized_llm_label(r)
        lgbm_p = r.lgbm_pred
        gt = "real" if r.ground_truth == "real" else "fake"

        if lgbm_p == llm_label:
            agree += 1
            continue

        pred_label, _conf, did_override = _replay_ensemble_row(
            r,
            linear_weights=linear_weights,
            linear_threshold=linear_threshold,
        )

        if did_override:
            override_accepted += 1
            if pred_label == gt:
                overrides_correct += 1
        else:
            override_blocked += 1
            if lgbm_p != gt:
                blocked_lgbm_wrong += 1

    return {
        "agree_with_lgbm": agree,
        "override_accepted": override_accepted,
        "override_blocked": override_blocked,
        "overrides_correct": overrides_correct,
        "blocked_but_lgbm_wrong": blocked_lgbm_wrong,
    }


def select_operating_point(
    rows: list[dict[str, Any]],
    *,
    primary: str = "macro_f1",
    max_override_rate: float | None = None,
) -> dict[str, Any] | None:
    """
    Pick best row by primary metric; optionally require override_rate <= cap.

    primary must be a key present on each row (e.g. macro_f1, f1, accuracy).
    """
    if not rows:
        return None

    def ok(r: dict[str, Any]) -> bool:
        if max_override_rate is None:
            return True
        return float(r.get("override_rate", 0.0)) <= max_override_rate + 1e-12

    feasible = [r for r in rows if ok(r)]
    pool = feasible if feasible else rows
    return max(pool, key=lambda r: float(r.get(primary, 0.0)))
