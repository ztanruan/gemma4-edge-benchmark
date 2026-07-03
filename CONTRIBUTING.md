# Contributing to Gemma 4 Edge Benchmark

Thanks for your interest! This repository is a focused benchmark harness — it
evaluates Gemma 4 26B-A4B served with vLLM on NVIDIA Jetson AGX Thor.
Contributions that keep the harness reproducible, deterministic, and
well-documented are very welcome.

## Dev setup

```bash
git clone https://github.com/ztanruan/gemma4-edge-benchmark
cd gemma4-edge-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

You do not need a Jetson (or any GPU) to work on the harness itself: the test
suite is fully offline and never contacts a vLLM endpoint.

## Before opening a PR

```bash
pre-commit install  # optional, but it runs ruff automatically
ruff check .        # lint (must pass)
ruff format --check .
pytest -q           # tests (must pass — offline, no endpoint needed)
```

Please add a test for any behavior change in the scoring, manifest parsing, or
maze/asset logic. Deterministic pieces (parsers, validators, simulators) should
stay covered without requiring live model runs.

## Scope

Good fits: new workload families with manifests and rubrics, scoring
robustness, better summaries and reports, support for additional
OpenAI-compatible backends, docs. Out of scope: bundling model weights or
datasets, and result changes that cannot be reproduced from the documented
deployment. If in doubt, open an issue first.

## Benchmark integrity

- Never commit raw run outputs, source datasets, or staged image corpora —
  `.gitignore` excludes `outputs/`, `data/source_datasets/`, and
  `data/image_corpora/` on purpose.
- Changes to scenario manifests or rubrics should explain what they measure
  and why in the accompanying use-case doc under `docs/use_cases/`.
- Published summary numbers must state the exact deployment conditions they
  were collected under (see the Deployment Under Test section of the README).

## Reporting bugs

Include the command you ran, the full error, your Python and vLLM versions,
and — for Jetson-specific issues — the JetPack/L4T version and container image.
Never paste server logs that contain private data.
