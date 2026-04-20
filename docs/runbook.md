# Jetson Thor Runbook

This document defines the recommended order for running the Gemma 4 vLLM benchmark on Jetson Thor.

## Goal

The run order is designed to catch configuration and metrics issues early, before you commit to the full workload and systems benchmark.

## Recommended Order

1. Generate assets at the intended image tier.
2. Validate the generated suite.
3. Run the API preflight check.
4. Run the smoke benchmark.
5. Run the full workload benchmark.
6. Run the full systems benchmark.
7. Summarize outputs and prepare review packets.

## 1. Generate Assets

Use the medium image tier for the benchmark baseline:

```bash
python3 scripts/generate_assets.py --image-tier medium
```

This generates:
- text corpora
- tool fixtures
- image subsets
- benchmark manifest
- scenario docs
- backend config and generation profiles

The image subsets now include:
- CIFAR-10 classification
- Caltech 256 classification
- multilingual text-in-image extraction

Image tier meanings:
- `small`: `200` images per dataset family
- `medium`: `500` images per dataset family
- `large`: `1000` images per dataset family, only when the selected source classes support it

## 2. Validate the Suite

```bash
python3 scripts/validate_suite.py
```

This checks:
- manifest integrity
- referenced text files exist
- referenced image files exist
- scenario docs exist

## 3. API Preflight

```bash
python3 scripts/preflight_smoke_test.py \
  --backend-config configs/backends/vllm.yaml
```

This validates the live server path before any benchmark run:
- `/v1/models`
- chat-aware `/tokenize`
- basic `/v1/chat/completions`
- structured JSON output with `response_format.json_schema`
- tool-call emission
- local image `file://` access

The preflight report now records explicit classified outcomes for:
- `reasoning_only_truncated`
- `parser_leak`
- `tool_call_leak`
- `structured_output_bypass`
- `length_truncated`
- `empty_stream_bug`

Use this to catch:
- wrong model served
- broken tokenizer path
- empty streaming issues
- broken structured-output path
- broken tool parser setup
- reasoning leaked into visible content
- raw tool-call protocol leaked into visible content
- structured-output grammar bypass under `thinking=false`
- true truncation versus parser-induced failure
- missing `--allowed-local-media-path`

## 4. Smoke Benchmark

Run the lightweight benchmark before the full benchmark:

```bash
python3 scripts/run_smoke_benchmark.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking both \
  --with-tegrastats
```

What it runs:
- workload smoke:
  - one representative scenario per workload family
  - `repeats=1`
  - per-scenario repeat overrides disabled
- systems smoke:
  - the tiny systems suite in `systems/smoke_manifest.yaml`

Why this matters:
- validates every major prompt path
- validates text, image, tool, and conversation flows
- validates the structured-output feature during the preflight stage that gates the smoke run
- validates metrics capture and SSE artifacts
- validates backend-profile metadata, prompt-token debug, and telemetry timeseries capture
- validates tegrastats collection if enabled
- validates systems experiment wiring without waiting for the full suite

Artifacts:
- workload smoke outputs under `outputs/smoke_runs/<timestamp>/workload/...`
- systems smoke outputs under `outputs/smoke_runs/<timestamp>/systems/...`
- a combined top-level `smoke_report.json`
- preflight writes per-check request payloads, completions, raw SSE, SSE timelines, and diagnosis JSON so a human or later review model can verify exactly what failed

## 5. Full Workload Benchmark

```bash
python3 scripts/run_benchmark.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking both \
  --repeats 3 \
  --warmup-count 2 \
  --with-tegrastats \
  --output-root outputs/live_runs
```

Notes:
- this runs the full workload suite
- image and text families are both included
- outputs are collected, not auto-scored

## 6. Full Systems Benchmark

```bash
python3 scripts/run_systems_benchmark.py \
  --backend-config configs/backends/vllm.yaml \
  --manifest systems/manifest.yaml \
  --with-tegrastats \
  --output-root outputs/systems_runs
```

This runs the full systems experiments:
- thermal soak
- concurrency scaling
- IO sweep
- ITL distribution
- energy efficiency
- KV cache saturation
- cold start
- prefix reuse

## 7. Summaries and Review Packets

Workload summary:

```bash
python3 scripts/summarize_results.py \
  --run-dir outputs/live_runs/<run_id>
```

Workload judge packets:

```bash
python3 scripts/prepare_judge_packets.py \
  --run-dir outputs/live_runs/<run_id>
```

Systems summary:

```bash
python3 scripts/summarize_systems_results.py \
  --run-dir outputs/systems_runs/<run_id>
```

The workload and systems CSV summaries now carry:
- backend profile and backend config path
- vision budget / `max_soft_tokens`
- prefix-caching mode
- multilingual metadata: `input_language`, `expected_output_language`, and `language_variant`
- prompt-token debug status and counts
- telemetry log and timeseries paths

For image-quality or image-extraction comparisons, repeat the run with:
- `configs/backends/vllm_image_280.yaml`
- `configs/backends/vllm_image_560.yaml`
- `configs/backends/vllm_image_1120.yaml`

## Fast Debug Variants

Workload dry-run:

```bash
python3 scripts/run_benchmark.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking both \
  --limit 2 \
  --dry-run
```

Smoke dry-run:

```bash
python3 scripts/run_smoke_benchmark.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking both \
  --dry-run
```

Systems smoke only:

```bash
python3 scripts/run_systems_benchmark.py \
  --backend-config configs/backends/vllm.yaml \
  --manifest systems/smoke_manifest.yaml \
  --with-tegrastats
```

## Backend Configs

- `configs/backends/vllm.yaml`: default baseline alias
- `configs/backends/vllm_baseline.yaml`: baseline benchmark config with prefix caching disabled
- `configs/backends/vllm_image.yaml`: image benchmark alias with pinned `max_soft_tokens=280`
- `configs/backends/vllm_image_280.yaml`: image benchmark variant at `280`
- `configs/backends/vllm_image_560.yaml`: image benchmark variant at `560`
- `configs/backends/vllm_image_1120.yaml`: image benchmark variant at `1120`
- `configs/backends/vllm_prefix_caching.yaml`: only for the `prefix_caching` systems experiment

Use the default baseline config for normal preflight, smoke, workload, and non-prefix systems runs. Switch to the image config for image-only family runs, and switch to the prefix-caching config only when the goal is to measure prefix reuse.

For image comparison runs, use the three explicit image configs and keep everything else the same so the only controlled variable is `max_soft_tokens`.

## Practical Rule

Do not launch the full benchmark until all three pass:
- `validate_suite.py`
- `preflight_smoke_test.py`
- `run_smoke_benchmark.py`
