# Analog Clock Time Reading Experiment

This experiment is part of the main image benchmark suite and stages a repo-local subset of the Kaggle clock dataset:

- local source copy: `data/source_datasets/time-image-datasetclassification`
- staged subset: `data/image_corpora/clock_time_subset`
- benchmark family: `image_clock_time_reading`

## Goal

Pass a single analog clock image to Gemma and require a closed-set time classification over all possible labels in the dataset.

The model must return JSON only:

```json
{"predicted_label":"5_40","confidence_band":"high","brief_reason":"Hour hand near 5, minute hand at 8."}
```

## Label Space

- `144` closed-set labels
- exact format: `H_MM`
- examples:
  - `1_00` means `1:00`
  - `5_40` means `5:40`
  - `12_05` means `12:05`

## Split and Tiers

This benchmark stages from the held-out `test` split only.

Clock-specific tier sizes:

- `small`: `2` samples per time label = `288` total
- `medium`: `7` samples per time label = `1008` total
- `large`: `10` samples per time label = `1440` total

These tier counts are intentionally different from CIFAR/Caltech because the held-out clock split only has `10` images per time label.

## Prompt Shape

- multimodal `/v1/chat/completions`
- image first, instruction second
- closed-set label list included in the prompt
- JSON-only response contract
- no auto-scoring during execution

## What Gets Logged

The benchmark logs the same evidence as the other image families:

- request payload
- visible response text
- reasoning text
- parsed tool data if present
- raw SSE
- SSE timeline
- token usage
- latency / TTFT
- Prometheus metrics deltas
- optional `tegrastats`
- scenario rubric doc with the correct time label

## Run

Generate assets first:

```bash
python3 scripts/generate_assets.py --image-tier medium
```

Run the clock family at the three pinned image budgets:

```bash
python3 scripts/run_benchmark.py \
  --backend-config configs/backends/vllm_image_280.yaml \
  --thinking both \
  --family image_clock_time_reading \
  --with-tegrastats
```

Repeat with:

- `configs/backends/vllm_image_560.yaml`
- `configs/backends/vllm_image_1120.yaml`

## Review

Use the generated scenario docs in `docs/scenarios/` plus [rubric.md](rubric.md) for later human or LLM evaluation.
