# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-02

First public release: the harness, workloads, and sanitized summaries behind
the Gemma 4 26B-A4B on Jetson AGX Thor evaluation.

### Added
- Phase 1 workload benchmark: 33 text, agent, multilingual, and image scenario
  families defined in `benchmarks/manifest.yaml`, with per-family rubrics.
- Main runner (`scripts/run_benchmark.py`) capturing request payloads, visible
  and reasoning text, tool calls, raw SSE streams, token usage, TTFT and
  end-to-end latency, Prometheus metric deltas, and optional Jetson telemetry.
- Preflight and smoke validation for chat, structured JSON, tool calling, and
  image requests in both `thinking=false` and `thinking=true` modes.
- Systems track (`scripts/run_systems_benchmark.py`): thermal soak, concurrency
  scaling, IO-length sweep, ITL distribution, energy efficiency, KV-cache
  saturation, cold start, and prefix caching.
- Deterministic maze-navigation and clock-time-reading research tracks with
  their own runners, validators, and rubrics.
- Asset staging pipeline (`scripts/generate_assets.py`) that builds benchmark
  corpora from local source datasets (CIFAR-10, Caltech-256, clock and
  multilingual OCR image sets), plus `scripts/validate_suite.py`.
- Rubric-based semantic scoring and summary generation
  (`scripts/summarize_results.py`, `scripts/summarize_systems_results.py`,
  `scripts/summarize_maze_navigation.py`), and LLM-judge packet preparation.
- Backend profiles for baseline, prefix caching, and image soft-token budgets
  (280/560/1120) under `configs/backends/`.
- Use-case scenario documentation for all twelve workload domains under
  `docs/use_cases/`, systems methodology in `docs/systems/`, and a long-form
  results write-up in `docs/gemma4_edge_blog_post.md`.
- Open-source scaffolding: MIT license, packaging (`pyproject.toml`), offline
  test suite, ruff lint/format, pre-commit hooks, CI, and issue templates.

[1.0.0]: https://github.com/ztanruan/gemma4-edge-benchmark/releases/tag/v1.0.0
