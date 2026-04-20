# Recorded Metrics

Each run record can include:

- model id, vLLM backend metadata, backend profile, pinned image soft-token budget, and Jetson system metadata
- scenario id, use case, family, mode, scenario_connectivity, execution_mode, and context_source
- thinking enabled or disabled
- repeat index and seed
- generation profile and sampling parameters
- message payload path and prompt token estimate from vLLM's chat-aware `/tokenize` endpoint
- image file paths for multimodal scenarios
- configured max context tokens and whether prompt truncation occurred
- per-turn latency, TTFT, token usage, finish reason, visible answer text, reasoning text, and tool calls
- raw SSE event paths, per-event timelines, and tokenize fallback diagnostics
- vLLM Prometheus metric deltas and derived server-side latency breakdowns
- optional tegrastats log path plus parsed telemetry summary
