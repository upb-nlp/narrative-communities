# narrative-map

Data-Driven Communities for Fake News Prediction and Narrative Analysis

This repository is the framework for a three-stage pipeline on Romanian fake news ([`upb-nlp/ro_fake_news`](https://huggingface.co/datasets/upb-nlp/ro_fake_news)):

1. **GraphRAG indexing** — build knowledge graphs and Leiden communities from news articles.
2. **Narrative analysis** — regenerate community reports with DSPy; measure narrative purity (NMI, ARI, blending).
3. **Prediction & evaluation** — LightGBM + community similarity + DSPy judge, fused with a `linear_v1` ensemble.

The repo contains **pipeline code only**. Runtime outputs (indexes, models, CSVs, metrics JSON) are written under `artifacts/` and are gitignored.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended)
- Local OpenAI-compatible **LLM** on port 8000 (DSPy judge, GraphRAG extraction, community reports)
- Local OpenAI-compatible **embedding** API on port 8001 (community similarity, GraphRAG embeddings)

## Installation

Run these commands **from this repo root** (`narrative-map/`). uv creates and uses a project-local `.venv` here — do not install into another project's virtualenv.

```bash
cd /path/to/narrative-map
uv sync
```

`uv sync` installs the package in editable mode (`[tool.uv] package = true`). You do not need a separate `uv pip install -e .` unless you disabled the uv package flag.

Verify:

```bash
uv run python -c "import narrative_map; print('ok')"
```

### Troubleshooting: `Failed to read aiohttp` / wrong `.venv`

If uv reports `Using Python ... environment at: .../claim-exploration/.venv` (or another project), you are not using this repo's environment. Either:

```bash
cd /path/to/narrative-map
uv sync
```

Or, if another project's venv is corrupted (missing `aiohttp` METADATA), recreate it there:

```bash
cd /path/to/that-other-project
rm -rf .venv
uv sync
```

## Artifact layout

```
artifacts/
├── graphrag/       # GraphRAG project root (settings.yaml, input/, output/)
├── communities/    # Per-run parquet outputs (copy from graphrag output after indexing)
├── reports/        # DSPy community report JSON
├── models/           # lightgbm_model.pkl, scaler.pkl, dspy_compiled.json
└── results/          # prediction CSVs, purity metrics JSON, summary cache
```

## Pipeline

### Stage 1 — GraphRAG indexing

```bash
uv run python -m narrative_map.indexing.server
# Resume after partial failure:
uv run python -m narrative_map.indexing.server --resume-from generate_text_embeddings
```

Copy GraphRAG output parquets into `artifacts/communities/<run_name>/` (e.g. `new_fake_news_36`, `new_real_news_36`).

### Stage 2a — DSPy community reports

```bash
uv run python -m narrative_map.community_reports.run \
  --data-dir artifacts/communities/new_fake_news_36/ \
  --write-parquet

uv run python -m narrative_map.community_reports.run \
  --data-dir artifacts/communities/new_real_news_36/ \
  --write-parquet
```

### Stage 2b — Narrative purity analysis (metrics only)

```bash
uv run python scripts/analyze_communities.py \
  --graph-dir artifacts/communities/new_all_qwen_36/
```

Writes JSON metrics to `artifacts/results/purity/global_metrics.json`.

### Stage 3a — Train LightGBM

```bash
uv run python scripts/train_classifier.py --tune --trials 30
```

### Stage 3b — Compile DSPy judge (once)

```bash
uv run python -m narrative_map.prediction.dspy_judge --compile
```

### Stage 3c — Run evaluation

```bash
uv run python scripts/run_evaluation.py --split test
```

Useful ablation flags:

- `--no-communities` — LightGBM + judge only
- `--no-llm` — LightGBM + community similarity only
- `--linear-w W_LGBM W_LLM W_SIM` and `--linear-threshold` — ensemble tuning

### Stage 3d — Offline ensemble tuning (no LLM calls)

```bash
uv run python scripts/eval_ensemble_offline.py \
  --csv artifacts/results/prediction_test.csv \
  --linear-w 0.05 0.30 0.65 \
  --linear-threshold 0.6

uv run python scripts/sweep_linear_v1.py \
  --tune-csv artifacts/results/prediction_validation.csv \
  --weight-step 0.05 \
  --threshold-points 33 \
  --out-csv artifacts/results/sweep_val.csv \
  --primary macro_f1
```

## Package layout

```
src/narrative_map/
├── indexing/           # GraphRAG config + ingestion
├── community_reports/  # DSPy report regeneration
├── analysis/           # Narrative purity metrics
├── prediction/         # Classifier, similarity, judge, ensemble, predict
└── evaluation/         # Threshold sweep helpers
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
