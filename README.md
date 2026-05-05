# HAII — Human-AI Interaction Investigation

Small experimental framework for measuring evaluation differences and potential bias
across several LLM providers (OpenAI, Anthropic / Claude, and xAI / Grok).

---

## Purpose

This repository runs controlled prompts across multiple model providers to evaluate
how model outputs vary by identity cues and writing style. It collects structured
JSON responses, saves raw outputs, and produces summary tables and plots.

## Repository layout

- `run_experiment.py` — Main multi-provider experiment runner. Loads `prompts.csv`,
  queries configured providers, and writes `outputs/generations.jsonl`.
- `run_experiment_openAI_only.py` — A simplified runner for only OpenAI (kept
  for convenience).
- `analyze_results.py` — Loads `outputs/generations.jsonl`, flattens records,
  computes summaries, and creates figures in `outputs/figures/`.
- `prompts.csv` — Experimental prompts (expected to contain exactly 9 prompts
  in the current setup).
- `outputs/` — Output folder (JSONL, CSV, figures, and report).

## Key defaults and safeguards

- The experiment expects exactly 9 prompts by default. `run_experiment.py` will
  raise an error if `prompts.csv` does not contain exactly 9 rows. This is
  controlled by `EXPECTED_PROMPT_COUNT`.
- Replicates per prompt/provider are controlled by `N_REPLICATES` (default 10).
  With 3 providers this gives `9 * 3 * 10 = 270` runs total.
- The runner persists raw responses for failed parses to aid debugging and now
  includes a robust JSON-extraction fallback for Anthropic outputs.

## Requirements

Install dependencies in a virtual environment. Typical packages used by these
scripts include:

```bash
python -m venv venv
source venv/bin/activate
pip install openai anthropic pandas matplotlib scipy python-dotenv
```

(If you use `requirements.txt` in your workflow, add the packages above there.)

## Configuration

The scripts read API keys from environment variables. Create a `.env` file or
export them in your shell before running:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="claude-..."
export XAI_API_KEY="xai-..."
```

Or add a `.env` file with the same variables — the code attempts to load it via
`python-dotenv` if present.

## Running the experiment

1. Verify `prompts.csv` contains the 9 prompts you intend to run.
2. Ensure your venv is active and API keys are available.
3. Run the multi-provider experiment:

```bash
python run_experiment.py
```

This prints a startup summary showing providers detected, replicates per
prompt/provider, runs per provider, and expected total runs. Progress is printed
as each run completes and results are appended to `outputs/generations.jsonl`.

If you prefer to run only OpenAI, use:

```bash
python run_experiment_openAI_only.py
```

## Analysis

After running the experiment, create the summary CSVs and figures by running:

```bash
python analyze_results.py
```

Outputs created by the analysis script:

- `outputs/generations_flat.csv` — flattened per-run table.
- `outputs/condition_summary.csv` — aggregated condition-level summary.
- `outputs/summary_report.txt` — plain-text findings and statistics.
- `outputs/figures/` — PNG charts (`avg_raise_by_condition_and_model.png`,
  `raise_yes_rate_by_condition_and_model.png`, `score_heatmap_by_condition_and_model.png`, etc.).


