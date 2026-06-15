#!/usr/bin/env python3
"""
Replay linear_v1 ensemble on a saved ``prediction_results*.csv`` — no LLM/embeddings calls.

**Columns**

- Full exports: ``lgbm_pred``, ``lgbm_conf``, ``llm_decision`` (or ``llm_pred``), ``llm_conf``,
  ``fake_max_sim``, ``real_max_sim``, ``actual``.
- Minimal exports (e.g. judge-only): ``predicted``, ``confidence``, community sims, ``actual``.
  Missing LightGBM columns fall back to an uninformative prior (``fake``, ``0.5`` → P(real)=0.5);
  judge label/confidence are taken from ``predicted`` / ``confidence`` when LLM columns are absent.

Example::

    uv run python scripts/eval_ensemble_offline.py \\
      --csv prediction_results_qwen_35.csv \\
      --linear-w 0.05 0.30 0.65 --linear-threshold 0.6

    uv run python scripts/eval_ensemble_offline.py \\
      --csv prediction_results_QwQ-32B.csv \\
      --linear-w 0.05 0.30 0.65 --linear-threshold 0.6 \\
      --out-csv prediction_results_QwQ-32B_linear_v1.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path (so `uv run python scripts/eval_ensemble_offline.py` finds the package).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from narrative_map.prediction.ensemble import ensemble_linear_v1


def _norm_llm(llm_decision: str, lgbm_pred: str) -> str:
    ll = (llm_decision or "").strip().lower()
    if ll not in ("fake", "real"):
        return (lgbm_pred or "fake").strip().lower()
    return ll


def _row_linear_inputs(
    row: pd.Series,
    *,
    warned: dict[str, bool],
) -> tuple[str, float, str, float, dict[str, float]]:
    """Map a CSV row to (lgbm_pred, lgbm_conf, llm_label, llm_conf, sim_stats)."""
    raw_lp = row.get("lgbm_pred", float("nan"))
    if pd.notna(raw_lp) and str(raw_lp).strip():
        lgbm = str(raw_lp).strip().lower()
        if lgbm not in ("fake", "real"):
            lgbm = "fake"
        lgbm_c = float(row.get("lgbm_conf", 0.5))
    else:
        if not warned.get("lgbm_fallback"):
            print(
                "  Note: missing lgbm_pred — using uninformative LightGBM "
                "(fake, conf=0.5 → P(real)=0.5). Export a full predict CSV for true LGBM terms.",
                file=sys.stderr,
                flush=True,
            )
            warned["lgbm_fallback"] = True
        lgbm, lgbm_c = "fake", 0.5

    raw_llm = row.get("llm_decision", row.get("llm_pred", float("nan")))
    if pd.notna(raw_llm) and str(raw_llm).strip():
        llm_src = str(raw_llm)
    else:
        alt = row.get("predicted", "")
        llm_src = str(alt) if pd.notna(alt) else ""

    llm = _norm_llm(llm_src, lgbm)

    raw_lc = row.get("llm_conf", float("nan"))
    if pd.notna(raw_lc):
        llm_c = float(raw_lc)
    else:
        cf = row.get("confidence", 0.0)
        llm_c = float(cf) if pd.notna(cf) else 0.0

    sim = {
        "fake_max_sim": float(row.get("fake_max_sim", 0) or 0),
        "real_max_sim": float(row.get("real_max_sim", 0) or 0),
    }
    return lgbm, lgbm_c, llm, llm_c, sim


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "correct": float(sum(1 for a, b in zip(y_true, y_pred) if a == b)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": float(len(y_true)),
    }


def _metrics_research(y_true: list[int], y_pred: list[int]) -> dict[str, float | dict | list]:
    """Aggregate metrics plus confusion matrix and per-class PRF (fake=0, real=1)."""
    base = _metrics(y_true, y_pred)
    if not y_true:
        return {
            **base,
            "confusion_matrix": [[0, 0], [0, 0]],
            "confusion_labels": ["true_fake", "true_real"],
            "confusion_cols": ["pred_fake", "pred_real"],
            "per_class": {},
        }
    yt = y_true
    yp = y_pred
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    prec, rec, f1s, _sup = precision_recall_fscore_support(
        yt, yp, labels=[0, 1], zero_division=0
    )
    return {
        **base,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": ["true_fake", "true_real"],
        "confusion_cols": ["pred_fake", "pred_real"],
        "per_class": {
            "fake": {
                "precision": float(prec[0]),
                "recall": float(rec[0]),
                "f1": float(f1s[0]),
            },
            "real": {
                "precision": float(prec[1]),
                "recall": float(rec[1]),
                "f1": float(f1s[1]),
            },
        },
    }


def eval_linear_v1_research(
    df: pd.DataFrame,
    *,
    linear_weights: tuple[float, float, float],
    linear_threshold: float,
    warned: dict[str, bool],
) -> dict[str, float | dict | list]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for _, row in df.iterrows():
        act = str(row.get("actual", "")).strip().lower()
        if act not in ("fake", "real"):
            continue
        yt = 1 if act == "real" else 0
        lgbm, lgbm_c, llm, llm_c, sim = _row_linear_inputs(row, warned=warned)
        pred, _conf, _ov, _pr = ensemble_linear_v1(
            lgbm,
            lgbm_c,
            llm,
            llm_c,
            sim,
            weights=linear_weights,
            decision_threshold=linear_threshold,
        )
        y_true.append(yt)
        y_pred.append(1 if pred == "real" else 0)
    if not y_true:
        out = _metrics_research([], [])
        out["n"] = 0.0
        return out
    return _metrics_research(y_true, y_pred)


def add_linear_v1_columns(
    df: pd.DataFrame,
    *,
    linear_weights: tuple[float, float, float],
    linear_threshold: float,
    warned: dict[str, bool],
) -> pd.DataFrame:
    out = df.copy()
    preds: list[str] = []
    confs: list[float] = []
    p_reals: list[float] = []
    overrides: list[bool] = []
    for _, row in out.iterrows():
        lgbm, lgbm_c, llm, llm_c, sim = _row_linear_inputs(row, warned=warned)
        pred, conf, ov, pr = ensemble_linear_v1(
            lgbm,
            lgbm_c,
            llm,
            llm_c,
            sim,
            weights=linear_weights,
            decision_threshold=linear_threshold,
        )
        preds.append(pred)
        confs.append(conf)
        p_reals.append(pr)
        overrides.append(ov)
    out["linear_v1_pred"] = preds
    out["linear_v1_conf"] = confs
    out["linear_v1_p_real"] = p_reals
    out["linear_v1_override_lgbm"] = overrides
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline linear_v1 replay on a prediction results CSV.",
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--linear-w",
        nargs=3,
        type=float,
        metavar=("W_LGBM", "W_LLM", "W_SIM"),
        default=None,
    )
    p.add_argument("--linear-threshold", type=float, default=0.5)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write rows plus linear_v1_pred, linear_v1_conf, linear_v1_p_real, "
        "linear_v1_override_lgbm",
    )
    p.add_argument(
        "--primary",
        type=str,
        choices=("correct", "accuracy", "f1", "macro_f1"),
        default="correct",
        help="Metric reported as primary in summary (default: %(default)s)",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="If set, write metrics JSON + summary JSON",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write a single JSON file with metrics and CLI config",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.out_csv is None and "actual" not in df.columns:
        raise SystemExit(
            "CSV must contain 'actual', or pass --out-csv to only annotate predictions.",
        )

    lin_w: tuple[float, float, float]
    if args.linear_w is not None:
        lin_w = (args.linear_w[0], args.linear_w[1], args.linear_w[2])
    else:
        lin_w = (1 / 3, 1 / 3, 1 / 3)

    warned: dict[str, bool] = {}

    if args.out_csv is not None:
        annotated = add_linear_v1_columns(
            df,
            linear_weights=lin_w,
            linear_threshold=args.linear_threshold,
            warned=warned,
        )
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        annotated.to_csv(args.out_csv, index=False)
        print(f"  Wrote {args.out_csv} ({len(annotated)} rows)", flush=True)

    if "actual" in df.columns:
        m_full = eval_linear_v1_research(
            df,
            linear_weights=lin_w,
            linear_threshold=args.linear_threshold,
            warned=warned,
        )
        summary_metrics = {
            "correct": float(m_full["correct"]),
            "accuracy": float(m_full["accuracy"]),
            "f1": float(m_full["f1"]),
            "macro_f1": float(m_full["macro_f1"]),
            "n": float(m_full["n"]),
        }

        print(
            f"\nOffline linear_v1  ({len(df)} rows in CSV; metrics n={summary_metrics['n']:.0f})\n",
            flush=True,
        )
        print(f"  {'metric':<12} {'value':>10}")
        for k, v in summary_metrics.items():
            print(f"  {k:<12} {v:10.4f}")

        run_config = {
            "csv": str(args.csv.resolve()),
            "linear_weights": list(lin_w),
            "linear_threshold": args.linear_threshold,
            "primary": args.primary,
            "out_csv": str(args.out_csv.resolve()) if args.out_csv else None,
        }

        summary_payload = {
            "config": run_config,
            "linear_v1": m_full,
        }

        if args.summary_json:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(
                json.dumps(summary_payload, indent=2),
                encoding="utf-8",
            )
            print(f"\n  Wrote summary JSON: {args.summary_json}")

        if args.artifacts_dir:
            root = args.artifacts_dir
            root.mkdir(parents=True, exist_ok=True)
            (root / "linear_v1_metrics.json").write_text(
                json.dumps({"metrics": m_full, "config": run_config}, indent=2),
                encoding="utf-8",
            )
            (root / "ensemble_eval_summary.json").write_text(
                json.dumps(summary_payload, indent=2),
                encoding="utf-8",
            )
            print(f"  Wrote artifacts under: {root}")
    elif args.out_csv is not None:
        print("(Skipped metrics: no 'actual' column.)", flush=True)


if __name__ == "__main__":
    main()
