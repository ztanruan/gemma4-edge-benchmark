# Gemma 4 Edge Benchmark on Jetson Thor

This repository packages a reproducible `vLLM` benchmark harness for evaluating `Gemma 4 26B-A4B` on `NVIDIA Jetson AGX Thor` across practical edge workloads. The focus is not generic chat demos. It is grounded, inspectable evaluation for offline and low-connectivity scenarios where you care about correctness, latency, structured outputs, multimodal inputs, and full artifact capture.

The benchmark exercises:

- grounded document QA and summarization
- structured extraction and format compliance
- multilingual text and multilingual image-text extraction
- closed-set image classification
- tool-use and conversation handling
- safety and robustness families such as abstention, prompt injection resistance, and citation/source attribution
- deterministic control-style stress tests such as maze navigation
- analog clock time reading as a fine-grained visual precision task
- Jetson-specific systems measurements such as latency, throughput, telemetry, and thermal behavior

## Why This Repo Exists

Gemma 4 is a strong fit for edge AI only if it can hold up on real edge-shaped tasks:

- answering from local documents without internet access
- reading images and returning machine-consumable JSON
- staying stable when prompts get longer or more structured
- handling multilingual content and mixed operational contexts
- exposing artifacts that a human or a later LLM judge can review

This repo is designed to make those claims measurable. It captures raw responses, reasoning traces when enabled, token accounting, per-request latency, streaming timelines, and rubric-ready artifacts instead of relying on a single leaderboard number.

## Deployment Under Test

The public summary in this repository reflects runs collected on `April 17-18, 2026` under the following conditions:

- Device: `NVIDIA Jetson AGX Thor Developer Kit`
- Platform: `Linux 6.8.12-tegra-aarch64-with-glibc2.39`
- Serving stack: `vLLM` with the OpenAI-compatible `/v1/chat/completions` endpoint
- Model: `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`
- Modalities enabled in this deployment: `text` and `image`
- Audio: intentionally disabled in the deployed benchmark build
- Reasoning modes: `thinking=false` and `thinking=true`
- Standard workload context limit: `65,536` tokens
- Standard workload output cap: `2,096` tokens
- Baseline image soft-token budget: `280`
- Additional image configs included: `560` and `1120`
- Prefix caching: disabled for baseline correctness and latency runs, with a separate backend profile for prefix-caching experiments
- Prompt-token accounting: uses vLLM's chat-aware `/tokenize` path with structured `messages`, so counts reflect the actual Gemma 4 chat template instead of a flattened approximation

Prompt construction follows the Gemma 4 chat path used by the serving stack:

- structured chat messages rather than hand-flattened role text
- official-style Gemma 4 reasoning and tool parsers in vLLM
- schema-constrained JSON where structured output matters
- multimodal chat requests for image workloads

## What Is In Scope

- `Phase 1 workload benchmark`: 33 text, agent, multilingual, and image families
- `Preflight and smoke`: fast checks for chat, structured output, tool calling, and image requests in both thinking modes
- `Systems track`: thermal soak, concurrency scaling, IO-length sweep, ITL distribution, energy efficiency, KV-cache saturation, cold start, and prefix caching
- `Maze navigation`: deterministic multi-step control experiment with shortest-path ground truth
- `Clock time reading`: closed-set analog clock classification over 144 labels

## Selected Results

### Phase 1 workload benchmark

- Weighted across `9,932` scored records, semantic task-completion accuracy was `93.43%`.
- Overall by mode, semantic accuracy was `95.45%` with `thinking=false` and `91.40%` with `thinking=true`.
- Because the large image families dominate total volume, the more honest non-image view was lower: `82.28%` with `thinking=false` and `72.47%` with `thinking=true`.
- Closed-set vision was strong in the baseline `max_soft_tokens=280` run:
  - `image_classification_cifar10`: `91.60%` no-think, `90.60%` think
  - `image_classification_caltech256`: `99.60%` no-think, `99.93%` think
  - `image_text_extraction_multilingual`: `97.70%` no-think, `88.00%` think
- Several operational text families were very strong in this run. `multi_hop_reasoning`, `temporal_sequence_reasoning`, `prompt_injection_resistance`, `abstention_calibration`, `prioritization_triage`, `redaction_pii_awareness`, and the multilingual grounded families all reached `100%` in both modes.
- `structured_extraction` improved with reasoning: `90.00%` with `thinking=false` and `96.67%` with `thinking=true`.
- Reasoning was not a universal win. It materially increased latency and introduced reasoning-only truncation in several long or format-sensitive families.

Concrete latency examples from the scored phase-one run:

- `grounded_qa`: median `1.63s` with `thinking=false` vs `13.13s` with `thinking=true`
- `structured_extraction`: median `2.28s` vs `25.01s`
- `image_classification_caltech256`: median `1.45s` vs `7.26s`
- `image_text_extraction_multilingual`: median `1.95s` vs `10.25s`

### Clock time reading

- The analog clock benchmark is a deliberately fine-grained closed-set image task with `144` possible labels from `1_00` through `12_55`.
- In the completed `280`-token vision-budget run, the model achieved `13.5%` exact-match accuracy while still reaching `99.7%` JSON parse success.
- That result is useful in practice because it separates two failure modes:
  - contract-following was strong
  - fine-grained hand-position recognition remained difficult

### Maze navigation

- The maze benchmark is a deterministic sequential-control stress test, not a passive QA task.
- The model solved the `easy` corridor level optimally in both reasoning modes.
- Under the current prompt and output budget, the `medium` and `hard` levels were not completed.
- This makes maze navigation a good stretch test for future prompt, decoding, and controller-loop improvements, but it is not currently a strength case for this deployment.

## Methodology

- Every scenario is generated with a scenario document and a review rubric.
- Preflight checks validate both `thinking=false` and `thinking=true` for:
  - basic chat
  - structured JSON output
  - tool-call emission and tool follow-up
  - image requests
- Phase 1 captures full request and response artifacts, including:
  - request payloads
  - visible response text
  - reasoning text when available
  - tool calls
  - raw SSE event streams and timelines
  - token-usage data
  - TTFT and end-to-end latency
  - Prometheus metric deltas
  - optional Jetson telemetry
- Semantic scoring is rubric-based rather than transport-based:
  - exact-label matching for closed-set classification
  - exact or constrained field validation for structured JSON tasks
  - transcript-and-language checks for multilingual image-text extraction
  - required-fact and prohibited-action checks for prose families
- Raw outputs are intended for local inspection. The public repo keeps the reproducible code and sanitized summaries, while large generated datasets and raw run artifacts remain local-only by default.

## Repository Layout

- `benchmarks/manifest.yaml`: workload definitions and scenario families
- `configs/backends/`: backend profiles for baseline, prefix caching, and image budgets
- `configs/generation_profiles.yaml`: standardized generation settings
- `scripts/generate_assets.py`: stages local benchmark assets from source datasets
- `scripts/preflight_smoke_test.py`: preflight validation for chat, tools, structure, and image
- `scripts/run_smoke_benchmark.py`: one-sample smoke run across workload families
- `scripts/run_benchmark.py`: main workload runner
- `scripts/summarize_results.py`: workload summary generation
- `scripts/run_systems_benchmark.py`: systems benchmark runner
- `scripts/summarize_systems_results.py`: systems summary generation
- `scripts/run_maze_navigation.py`: deterministic maze experiment runner
- `scripts/run_maze_navigation_smoke.py`: maze smoke validation
- `scripts/summarize_maze_navigation.py`: maze run summarization
- `scripts/validate_suite.py`: validates generated benchmark assets
- `clock_time_reading/README.md`: clock experiment details and rubric
- `maze_navigation/README.md`: maze experiment details and rubric
- `docs/systems/README.md`: systems benchmark methodology

## Reproducing The Benchmark

This repo is intended to be run from the repository root.

1. Create an environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place local source datasets under `data/source_datasets/`.

Expected local dataset folders:

- `data/source_datasets/cifar-10-batches-py`
- `data/source_datasets/256_ObjectCategories`
- `data/source_datasets/time-image-datasetclassification`
- `data/source_datasets/multilingual-image-text-translation`

3. Generate staged benchmark assets.

```bash
python3 scripts/generate_assets.py --image-tier medium
python3 scripts/validate_suite.py
```

4. Run preflight and smoke validation before the full workload.

```bash
python3 scripts/preflight_smoke_test.py --backend-config configs/backends/vllm_baseline.yaml --thinking both
python3 scripts/run_smoke_benchmark.py --backend-config configs/backends/vllm_baseline.yaml --thinking both
```

5. Run the main phase-one workload benchmark.

```bash
python3 scripts/run_benchmark.py \
  --backend-config configs/backends/vllm_baseline.yaml \
  --thinking both \
  --repeats 3
```

6. Summarize results locally.

```bash
python3 scripts/summarize_results.py --run-dir outputs/live_runs/<run_id>
```

Additional entry points:

- Clock benchmark:

```bash
python3 scripts/run_benchmark.py \
  --backend-config configs/backends/vllm_image_280.yaml \
  --family image_clock_time_reading \
  --thinking both
```

- Maze benchmark:

```bash
python3 scripts/run_maze_navigation_smoke.py --backend-config configs/backends/vllm_baseline.yaml --thinking both
python3 scripts/run_maze_navigation.py --backend-config configs/backends/vllm_baseline.yaml --thinking both
```

- Systems benchmark:

```bash
python3 scripts/run_systems_benchmark.py \
  --backend-config configs/backends/vllm_baseline.yaml \
  --manifest systems/manifest.yaml
```

## Publish Policy

The public-facing repository is meant to contain:

- code
- configs
- manifests
- experiment docs
- sanitized summaries

The following are intentionally treated as local-only artifacts and are ignored by default:

- raw benchmark outputs under `outputs/`
- staged image corpora under `data/image_corpora/`
- original source datasets under `data/source_datasets/`
- generated scenario docs under `docs/scenarios/`

This keeps the public repo small, reproducible, and free of machine-specific traces such as absolute local paths, telemetry logs, and raw streaming artifacts.
