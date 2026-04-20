# Systems Benchmark Track

This track is separate from the workload benchmark.

The workload benchmark answers:
- how Gemma behaves on realistic edge tasks
- what responses, reasoning, and tool calls it produces
- what latency and telemetry look like per scenario

The systems benchmark answers:
- whether Jetson Thor can sustain the workload thermally
- how vLLM scales with concurrency
- how prompt length and output length affect latency
- whether streamed output is smooth
- what inference costs in power and energy terms
- how the server behaves under cache pressure, cold starts, and prefix reuse

The source of truth for experiment defaults is [systems/manifest.yaml](../../systems/manifest.yaml).

## Experiments

### `thermal_soak`

- Repeats a fixed medium-long request for a sustained window.
- Collects per-request latency, TTFT, token counts, throughput, telemetry, and a tegrastats time series.
- Main question: can Jetson Thor sustain this workload without unacceptable throughput loss as temperature rises?

### `concurrency_scaling`

- Runs simultaneous requests at `1, 2, 4, 8, 16` concurrency.
- Collects per-request latency and TTFT plus batch-level aggregate throughput.
- Main question: how many simultaneous users can this deployment handle before latency and queueing become unacceptable?

### `io_length_sweep`

- Sweeps controlled prompt lengths and output lengths.
- Covers short chat, short-in/long-out generation, and long-context grounded workloads.
- Main question: what is the latency profile for prefill-heavy vs decode-heavy requests?

### `itl_distribution`

- Uses per-event streaming timestamps to compute inter-token latency statistics.
- Tracks p50, p95, p99, max, and stall count.
- Main question: does streaming remain smooth, or does it stutter under decode load?

### `energy_efficiency`

- Runs representative short, medium, and long requests with per-request telemetry.
- Computes joules per request and joules per output token.
- Main question: what is the power cost of serving this model at the edge?

### `kv_cache_saturation`

- Launches overlapping long-context requests to push memory and queue pressure.
- Tracks queue time, TTFT, and KV-cache usage from vLLM metrics.
- Main question: what happens when the server approaches cache pressure on Jetson Thor?

### `cold_start`

- Measures the first request after attach, then after idle windows.
- Best run immediately after the server is launched.
- Main question: how expensive is first-user latency after startup or idle?

### `prefix_caching`

- Reuses a long shared prefix across many questions.
- Tracks TTFT, prefill time, and steady-state latency after the first request.
- Main question: does prefix reuse materially help this deployment on Jetson?
- Run this experiment with `configs/backends/vllm_prefix_caching.yaml`, not the baseline config.

## Response Review Sanity Checklist

These are systems tests, not correctness benchmarks, but responses are still archived. A later human reviewer can use this short checklist:

- Is the answer non-empty and coherent?
- Does the answer broadly follow the requested format?
- Does the answer avoid obvious hallucinated structure unrelated to the prompt?
- Does the answer remain stable across repeats, or does it degrade under load?
- Under thermal or concurrency stress, do later responses become truncated, repetitive, or malformed?

## Output Shape

The systems runner writes:

- `records.jsonl`: raw request, batch, and experiment-summary records
- `responses/`: full response text, reasoning, raw SSE, event timelines, and request payloads
- `telemetry/`: tegrastats logs and inferred time-series JSONL files
- `run_metadata.json`: run configuration and experiment result status
- `systems_summary.csv`: flattened request and batch records
- `systems_condition_summary.csv`: grouped aggregates by experiment and condition
