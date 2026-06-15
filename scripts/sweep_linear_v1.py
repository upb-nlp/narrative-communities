#!/usr/bin/env python3
"""
Offline grid sweep for ``linear_v1``: weights (on the 2-simplex) × ``linear_threshold``.

The three weights ``(w_lgbm, w_llm, w_sim)`` must be non-negative and sum to 1; this
script enumerates ``(w_lgbm, w_llm)`` on a grid and sets ``w_sim = 1 - w_lgbm - w_llm``
(skips invalid corners where ``w_sim < 0``).

For each weight triple and each ``linear_threshold``, replays ``linear_v1`` on a
saved ``prediction_results*.csv`` (same columns as ``eval_ensemble_offline.py``).

Tune on validation, optionally report on held-out test::

    uv run python scripts/sweep_linear_v1.py \\
      --tune-csv prediction_results_validation.csv \\
      --eval-csv prediction_results_test.csv \\
      --weight-step 0.05 --threshold-points 33 \\
      --out-csv sweep_linear_v1/val_grid.csv \\
      --primary macro_f1

Legacy (single CSV)::

    uv run python scripts/sweep_linear_v1.py \\
      --csv prediction_results_qwen_35.csv \\
      --weight-step 0.1 \\
      --threshold-min 0.1 --threshold-max 0.9 --threshold-points 17 \\
      --out-csv sweep_linear_v1/grid.csv \\
      --primary macro_f1

After the run, the script prints **three** global optima: **F1** (binary, positive
class = ``real``), **macro_F1**, and **accuracy**, and writes ``*_bests_by_metric.csv``.

Optional 2D heatmap (max over ``linear_threshold`` at each ``(w_lgbm, w_llm)``)::

    uv run python scripts/sweep_linear_v1.py \\
      --csv prediction_results_qwen_35.csv \\
      --weight-step 0.05 \\
      --threshold-points 33 \\
      --out-csv sweep_linear_v1/grid.csv \\
      --heatmap sweep_linear_v1/best_f1_simplex.png \\
      --heatmap-metric f1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from narrative_map.prediction.ensemble import ensemble_forward


def _norm_llm(llm_decision: str, lgbm_pred: str) -> str:
    ll = (llm_decision or "").strip().lower()
    if ll not in ("fake", "real"):
        return (lgbm_pred or "fake").strip().lower()
    return ll


def eval_linear(
    df: pd.DataFrame,
    *,
    linear_weights: tuple[float, float, float],
    linear_threshold: float,
) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for _, row in df.iterrows():
        act = str(row.get("actual", "")).strip().lower()
        if act not in ("fake", "real"):
            continue
        yt = 1 if act == "real" else 0
        lgbm = str(row.get("lgbm_pred", "fake")).strip().lower()
        lgbm_c = float(row.get("lgbm_conf", 0.5))
        llm_raw = row.get("llm_decision", row.get("llm_pred", ""))
        llm = _norm_llm(str(llm_raw), lgbm)
        llm_c = float(row.get("llm_conf", 0.0))
        sim = {
            "fake_max_sim": float(row.get("fake_max_sim", 0) or 0),
            "real_max_sim": float(row.get("real_max_sim", 0) or 0),
        }
        pred, _c, _ov = ensemble_forward(
            lgbm,
            lgbm_c,
            llm,
            llm_c,
            sim,
            linear_weights=linear_weights,
            linear_threshold=linear_threshold,
        )
        y_true.append(yt)
        y_pred.append(1 if pred == "real" else 0)
    return y_true, y_pred


def iter_weight_triples(step: float) -> list[tuple[float, float, float]]:
    """Non-negative weights on the simplex: w_sim = 1 - w_lgbm - w_llm."""
    if step <= 0:
        raise ValueError("weight-step must be positive")
    out: list[tuple[float, float, float]] = []
    w1_vals = np.arange(0.0, 1.0 + step * 0.5001, step, dtype=np.float64)
    for w1 in w1_vals:
        w1 = float(min(1.0, max(0.0, w1)))
        w2_max = 1.0 - w1 + 1e-12
        w2_vals = np.arange(0.0, w2_max + step * 0.5001, step, dtype=np.float64)
        for w2 in w2_vals:
            w2 = float(min(w2_max, max(0.0, w2)))
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = float(max(0.0, w3))
            out.append((w1, w2, w3))
    return out


def metric_scores(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0, "correct": 0.0, "n": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "correct": float(sum(1 for a, b in zip(y_true, y_pred) if a == b)),
        "n": float(len(y_true)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Grid sweep: linear_v1 (w_lgbm, w_llm, w_sim) × linear_threshold on a CSV.",
    )
    p.add_argument(
        "--tune-csv",
        type=Path,
        default=None,
        help="Validation (or tune) CSV for grid search",
    )
    p.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Optional held-out test CSV: replay best (w, τ) and write *_test_eval.csv",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Alias for --tune-csv (backward compatible)",
    )
    p.add_argument("--weight-step", type=float, default=0.1, help="Grid step for w_lgbm and w_llm")
    p.add_argument("--threshold-min", type=float, default=0.05)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--threshold-points", type=int, default=19)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument(
        "--heatmap",
        type=Path,
        default=None,
        help="Optional PNG: best metric vs (w_lgbm, w_llm) after max over τ (see --heatmap-metric)",
    )
    p.add_argument(
        "--heatmap-metric",
        type=str,
        choices=("macro_f1", "f1", "accuracy"),
        default=None,
        help="Which metric to maximize over τ for the heatmap (default: same as --primary)",
    )
    p.add_argument(
        "--primary",
        type=str,
        choices=("macro_f1", "f1", "accuracy"),
        default="macro_f1",
        help="Legacy: used for heatmap if --heatmap-metric omitted; all three bests are always printed.",
    )
    args = p.parse_args()
    heatmap_metric = args.heatmap_metric or args.primary

    tune_path = args.tune_csv or args.csv
    if tune_path is None:
        raise SystemExit("Provide --tune-csv or --csv (validation predictions CSV).")

    df = pd.read_csv(tune_path)
    if "actual" not in df.columns:
        raise SystemExit("CSV must contain an 'actual' column (real/fake).")
    print(f"Tuning on {tune_path.resolve()} ({len(df)} rows)")

    triples = iter_weight_triples(float(args.weight_step))
    thresholds = np.linspace(
        float(args.threshold_min),
        float(args.threshold_max),
        int(args.threshold_points),
        dtype=np.float64,
    )

    rows: list[dict[str, float]] = []
    # Per (w_lgbm, w_llm): best value and τ for each metric (max over τ; ties → smaller τ)
    best_at_w: dict[tuple[float, float], dict[str, float]] = {}

    def _better(new_val: float, new_tau: float, old_val: float, old_tau: float) -> bool:
        if new_val > old_val + 1e-12:
            return True
        if abs(new_val - old_val) <= 1e-12 and new_tau < old_tau:
            return True
        return False

    for w1, w2, w3 in triples:
        wkey = (round(w1, 8), round(w2, 8))
        best = {
            "macro_f1": float("-inf"),
            "tau_macro": float(args.threshold_min),
            "f1": float("-inf"),
            "tau_f1": float(args.threshold_min),
            "accuracy": float("-inf"),
            "tau_acc": float(args.threshold_min),
            "w_sim": w3,
        }
        for tau in thresholds:
            yt, yp = eval_linear(df, linear_weights=(w1, w2, w3), linear_threshold=float(tau))
            m = metric_scores(yt, yp)
            rows.append(
                {
                    "w_lgbm": w1,
                    "w_llm": w2,
                    "w_sim": w3,
                    "linear_threshold": float(tau),
                    "accuracy": m["accuracy"],
                    "f1": m["f1"],
                    "macro_f1": m["macro_f1"],
                    "correct": m["correct"],
                    "n": m["n"],
                }
            )
            t = float(tau)
            if _better(m["macro_f1"], t, best["macro_f1"], best["tau_macro"]):
                best["macro_f1"] = float(m["macro_f1"])
                best["tau_macro"] = t
            if _better(m["f1"], t, best["f1"], best["tau_f1"]):
                best["f1"] = float(m["f1"])
                best["tau_f1"] = t
            if _better(m["accuracy"], t, best["accuracy"], best["tau_acc"]):
                best["accuracy"] = float(m["accuracy"])
                best["tau_acc"] = t
        best_at_w[wkey] = best

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    def _best_row(metric: str) -> pd.Series:
        return out_df.iloc[int(out_df[metric].idxmax())]

    print(f"Wrote {args.out_csv.resolve()}  ({len(out_df)} rows, grid={len(triples)}×{len(thresholds)})")

    summary_rows: list[dict[str, object]] = []
    for metric, title in (
        ("f1", "F1 (binary, pos=real)"),
        ("macro_f1", "macro_F1"),
        ("accuracy", "accuracy"),
    ):
        br = _best_row(metric)
        summary_rows.append(
            {
                "metric": metric,
                "title": title,
                "w_lgbm": float(br["w_lgbm"]),
                "w_llm": float(br["w_llm"]),
                "w_sim": float(br["w_sim"]),
                "linear_threshold": float(br["linear_threshold"]),
                "f1": float(br["f1"]),
                "macro_f1": float(br["macro_f1"]),
                "accuracy": float(br["accuracy"]),
                "correct": int(br["correct"]),
                "n": int(br["n"]),
            }
        )
        print(
            f"  Best {title}: {float(br[metric]):.4f}  "
            f"w=({br['w_lgbm']:.4f},{br['w_llm']:.4f},{br['w_sim']:.4f})  "
            f"linear_threshold={br['linear_threshold']:.4f}  "
            f"correct={int(br['correct'])}/{int(br['n'])}  "
            f"(at that τ: f1={br['f1']:.4f}, macro_f1={br['macro_f1']:.4f}, acc={br['accuracy']:.4f})"
        )

    bests_path = args.out_csv.with_name(args.out_csv.stem + "_bests_by_metric.csv")
    pd.DataFrame(summary_rows).to_csv(bests_path, index=False)
    print(f"Wrote {bests_path.resolve()}")

    if args.eval_csv is not None:
        eval_df = pd.read_csv(args.eval_csv)
        if "actual" not in eval_df.columns:
            raise SystemExit("--eval-csv must contain an 'actual' column (real/fake).")
        print(f"\nHeld-out eval on {args.eval_csv.resolve()} ({len(eval_df)} rows)")
        test_summary: list[dict[str, object]] = []
        for metric, title in (
            ("f1", "F1 (binary, pos=real)"),
            ("macro_f1", "macro_F1"),
            ("accuracy", "accuracy"),
        ):
            br = _best_row(metric)
            w = (float(br["w_lgbm"]), float(br["w_llm"]), float(br["w_sim"]))
            tau = float(br["linear_threshold"])
            yt, yp = eval_linear(
                eval_df, linear_weights=w, linear_threshold=tau,
            )
            m = metric_scores(yt, yp)
            test_summary.append(
                {
                    "metric": metric,
                    "title": title,
                    "w_lgbm": w[0],
                    "w_llm": w[1],
                    "w_sim": w[2],
                    "linear_threshold": tau,
                    "f1": m["f1"],
                    "macro_f1": m["macro_f1"],
                    "accuracy": m["accuracy"],
                    "correct": int(m["correct"]),
                    "n": int(m["n"]),
                }
            )
            print(
                f"  Test @ best {title} (from val): {float(m[metric]):.4f}  "
                f"w=({w[0]:.4f},{w[1]:.4f},{w[2]:.4f})  "
                f"linear_threshold={tau:.4f}  "
                f"correct={int(m['correct'])}/{int(m['n'])}"
            )
        test_path = args.out_csv.with_name(args.out_csv.stem + "_test_eval.csv")
        pd.DataFrame(test_summary).to_csv(test_path, index=False)
        print(f"Wrote {test_path.resolve()}")

    if args.heatmap is not None:
        print("Warning: --heatmap is not supported (plotting removed from paper framework).")


if __name__ == "__main__":
    main()
