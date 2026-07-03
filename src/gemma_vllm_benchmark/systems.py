from __future__ import annotations

import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .runner import (
    _backend_record_fields,
    _chat_template_kwargs,
    _completion_status,
    _derive_server_stats,
    _metric_average,
    _write_json,
    _write_text,
    collect_run_metadata,
    ensure_run_dirs,
    load_yaml,
    metrics_delta,
    request_chat_completion,
    scrape_metrics,
    tokenize_chat_messages_detailed,
    update_run_metadata,
    verify_model,
    write_record,
)
from .tegrastats import (
    TegraStatsSession,
    summarize_tegrastats_log,
    write_tegrastats_timeseries,
)

SYSTEMS_SYSTEM_PROMPT = (
    "You are Gemma 4 running in a vLLM systems benchmark on Jetson Thor. "
    "Follow the task exactly. Keep the answer concise, factual, and stable."
)


def _tokenize_debug_summary(detail: dict[str, Any]) -> dict[str, Any]:
    raw = detail.get("raw") or {}
    summary = {
        "request_type": detail.get("request_type"),
        "status_code": detail.get("status_code"),
        "ok": detail.get("ok"),
        "count": detail.get("count"),
        "error_type": detail.get("error_type"),
        "error": detail.get("error"),
    }
    if isinstance(raw, dict) and raw.get("max_model_len") is not None:
        summary["max_model_len"] = raw.get("max_model_len")
    return summary


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = index - lower
    interpolated = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(interpolated, 3)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _itl_stats(output_event_offsets_ms: list[float]) -> dict[str, Any]:
    if len(output_event_offsets_ms) < 2:
        return {
            "output_event_count": len(output_event_offsets_ms),
            "itl_count": 0,
            "itl_median_ms": None,
            "itl_p50_ms": None,
            "itl_p95_ms": None,
            "itl_p99_ms": None,
            "itl_max_ms": None,
            "stall_threshold_ms": None,
            "stall_count": 0,
        }
    intervals = [
        round(output_event_offsets_ms[index] - output_event_offsets_ms[index - 1], 3)
        for index in range(1, len(output_event_offsets_ms))
    ]
    median = statistics.median(intervals)
    stall_threshold = median * 3.0 if median > 0 else None
    stall_count = 0
    if stall_threshold is not None:
        stall_count = sum(1 for value in intervals if value > stall_threshold)
    return {
        "output_event_count": len(output_event_offsets_ms),
        "itl_count": len(intervals),
        "itl_median_ms": round(median, 3),
        "itl_p50_ms": _percentile(intervals, 0.50),
        "itl_p95_ms": _percentile(intervals, 0.95),
        "itl_p99_ms": _percentile(intervals, 0.99),
        "itl_max_ms": round(max(intervals), 3),
        "stall_threshold_ms": round(stall_threshold, 3) if stall_threshold is not None else None,
        "stall_count": stall_count,
    }


def _compute_joules(
    telemetry_summary: dict[str, Any] | None, elapsed_ms: float | None
) -> float | None:
    if not telemetry_summary or elapsed_ms is None:
        return None
    power_current_mw_avg = telemetry_summary.get("power_current_mw_avg")
    if power_current_mw_avg is None:
        return None
    return round((power_current_mw_avg * (elapsed_ms / 1000.0)) / 1000.0, 6)


def _render_reference_block(block_index: int) -> str:
    return (
        f"### Reference Block {block_index}\n"
        f"- Observation: edge node shard {block_index:04d} reported stable service operation before a transient queue spike.\n"
        f"- Action note: preserve evidence, summarize deltas, and state uncertainty instead of inventing missing facts.\n"
        f"- Local constraints: keep the response concise, do not expose hidden reasoning, and prefer exact values when listed.\n"
        f"- Diagnostic detail: request group {block_index:04d} included mixed prompt sizes, streaming output, and tool-free text generation.\n"
        f"- Follow-up: explain the likely cause, reference the supplied context, and recommend the minimum safe next step.\n"
    )


def _render_synthetic_user_prompt(
    *,
    question: str,
    context_blocks: list[str],
    request_label: str | None,
    preserve_prefix: bool,
) -> str:
    lines = [
        "Synthetic benchmark dossier for Jetson Thor + Gemma 4 via vLLM.",
        "Use only the supplied dossier and answer the question directly.",
        "",
    ]
    if request_label and not preserve_prefix:
        lines.extend([f"Request label: {request_label}", ""])
    lines.append("Dossier:")
    lines.append("")
    lines.extend(context_blocks)
    lines.extend(["", "Task:", question])
    if request_label and preserve_prefix:
        lines.append(f"Unique request label: {request_label}")
    return "\n".join(lines).strip()


def _build_synthetic_messages(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    target_prompt_tokens: int,
    question: str,
    thinking_enabled: bool,
    request_label: str | None = None,
    preserve_prefix: bool = False,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    def _messages_for_blocks(blocks: list[str]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEMS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _render_synthetic_user_prompt(
                    question=question,
                    context_blocks=blocks,
                    request_label=request_label,
                    preserve_prefix=preserve_prefix,
                ),
            },
        ]

    base_blocks = [
        "### Benchmark Context\n"
        "- This request is part of a controlled systems benchmark.\n"
        "- The device is Jetson Thor running vLLM chat serving.\n"
        "- Responses should remain concise and grounded in the provided dossier.\n"
    ]
    messages = _messages_for_blocks(base_blocks)
    detail = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs=_chat_template_kwargs(thinking_enabled),
    )
    if not detail.get("ok") or detail.get("count") is None:
        raise RuntimeError(
            f"Unable to tokenize chat messages through /tokenize for systems benchmark: "
            f"{detail.get('error_type')}: {detail.get('error')}"
        )
    base_count = int(detail["count"])
    if base_count >= target_prompt_tokens:
        return messages, base_count, _tokenize_debug_summary(detail)

    sample_messages = _messages_for_blocks(base_blocks + [_render_reference_block(1)])
    sample_detail = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=sample_messages,
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs=_chat_template_kwargs(thinking_enabled),
    )
    if not sample_detail.get("ok") or sample_detail.get("count") is None:
        raise RuntimeError(
            f"Unable to estimate synthetic block token cost through /tokenize: "
            f"{sample_detail.get('error_type')}: {sample_detail.get('error')}"
        )
    block_cost = max(1, int(sample_detail["count"]) - base_count)
    blocks_needed = max(1, math.ceil((target_prompt_tokens - base_count) / block_cost))
    blocks = base_blocks + [_render_reference_block(index) for index in range(1, blocks_needed + 1)]
    detail = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=_messages_for_blocks(blocks),
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs=_chat_template_kwargs(thinking_enabled),
    )
    if not detail.get("ok") or detail.get("count") is None:
        raise RuntimeError(
            f"Unable to tokenize full synthetic prompt through /tokenize: "
            f"{detail.get('error_type')}: {detail.get('error')}"
        )
    count = int(detail["count"])
    grow_index = blocks_needed + 1
    attempts = 0
    while count < target_prompt_tokens and attempts < 8:
        grow_by = max(1, math.ceil((target_prompt_tokens - count) / block_cost))
        blocks.extend(
            _render_reference_block(index) for index in range(grow_index, grow_index + grow_by)
        )
        grow_index += grow_by
        detail = tokenize_chat_messages_detailed(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=_messages_for_blocks(blocks),
            add_special_tokens=True,
            add_generation_prompt=True,
            continue_final_message=False,
            chat_template_kwargs=_chat_template_kwargs(thinking_enabled),
        )
        if not detail.get("ok") or detail.get("count") is None:
            raise RuntimeError(
                f"Unable to expand synthetic prompt through /tokenize: "
                f"{detail.get('error_type')}: {detail.get('error')}"
            )
        count = int(detail["count"])
        attempts += 1

    attempts = 0
    while len(blocks) > 1 and count > target_prompt_tokens + block_cost and attempts < 12:
        trim_by = max(1, math.ceil((count - target_prompt_tokens) / block_cost))
        trim_to = max(1, len(blocks) - trim_by)
        blocks = blocks[:trim_to]
        detail = tokenize_chat_messages_detailed(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=_messages_for_blocks(blocks),
            add_special_tokens=True,
            add_generation_prompt=True,
            continue_final_message=False,
            chat_template_kwargs=_chat_template_kwargs(thinking_enabled),
        )
        if not detail.get("ok") or detail.get("count") is None:
            raise RuntimeError(
                f"Unable to trim synthetic prompt through /tokenize: "
                f"{detail.get('error_type')}: {detail.get('error')}"
            )
        count = int(detail["count"])
        attempts += 1

    messages = _messages_for_blocks(blocks)
    return messages, count, _tokenize_debug_summary(detail)


def _write_completion_artifacts(
    response_base: Path,
    completion: dict[str, Any],
    prompt_token_debug: dict[str, Any] | None,
) -> dict[str, str | None]:
    response_base.mkdir(parents=True, exist_ok=True)
    response_path = response_base / "response.txt"
    reasoning_path = response_base / "reasoning.txt"
    tool_calls_path = response_base / "tool_calls.json"
    raw_events_path = response_base / "response_events.sse"
    event_timeline_path = response_base / "response_events_timeline.jsonl"
    payload_path = response_base / "request_payload.json"
    fallback_debug_path = response_base / "completion_token_fallback_debug.json"
    prompt_token_debug_path = response_base / "prompt_token_debug.json"

    _write_text(response_path, completion["content"])
    _write_text(reasoning_path, completion["reasoning"])
    _write_json(tool_calls_path, completion["tool_calls"])
    _write_text(raw_events_path, "\n".join(completion["raw_events"]))
    _write_text(
        event_timeline_path,
        "\n".join(json.dumps(item) for item in completion["raw_event_records"]),
    )
    _write_json(payload_path, completion["request_payload"])
    if prompt_token_debug is not None:
        _write_json(prompt_token_debug_path, prompt_token_debug)
    if completion.get("completion_token_fallback_debug") is not None:
        _write_json(fallback_debug_path, completion["completion_token_fallback_debug"])
        fallback_debug_path_str: str | None = str(fallback_debug_path)
    else:
        fallback_debug_path_str = None

    return {
        "response_path": str(response_path),
        "reasoning_path": str(reasoning_path),
        "tool_calls_path": str(tool_calls_path),
        "raw_events_path": str(raw_events_path),
        "raw_event_timeline_path": str(event_timeline_path),
        "request_payload_path": str(payload_path),
        "prompt_token_debug_path": str(prompt_token_debug_path)
        if prompt_token_debug is not None
        else None,
        "completion_token_fallback_debug_path": fallback_debug_path_str,
    }


def _build_request_record(
    *,
    backend_fields: dict[str, Any],
    experiment_id: str,
    experiment_kind: str,
    phase: str | None,
    response_id: str,
    request_label: str | None,
    target_prompt_tokens: int,
    actual_prompt_tokens: int | None,
    prompt_token_debug: dict[str, Any] | None,
    generation: dict[str, Any],
    thinking_enabled: bool,
    seed: int | None,
    completion: dict[str, Any],
    metric_values: dict[str, Any] | None,
    telemetry_summary: dict[str, Any] | None,
    telemetry_path: str | None,
    telemetry_timeseries_path: str | None,
    artifact_paths: dict[str, str | None],
    batch_id: str | None = None,
    batch_concurrency: int | None = None,
    case_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    server_stats = _derive_server_stats(completion, metric_values)
    effective_ttft_ms = (
        completion["ttft_ms"]
        if completion["ttft_ms"] is not None
        else server_stats.get("server_ttft_ms")
    )
    completion_tokens = (completion.get("usage") or {}).get("completion_tokens")
    joules_per_request = _compute_joules(telemetry_summary, completion["latency_ms"])
    joules_per_output_token = None
    if joules_per_request is not None and completion_tokens:
        joules_per_output_token = round(joules_per_request / completion_tokens, 6)
    record = {
        **backend_fields,
        "record_type": "request",
        "experiment_id": experiment_id,
        "experiment_kind": experiment_kind,
        "phase": phase,
        "case_id": case_id,
        "batch_id": batch_id,
        "batch_concurrency": batch_concurrency,
        "response_id": response_id,
        "request_label": request_label,
        "thinking_enabled": thinking_enabled,
        "seed": seed,
        "target_prompt_tokens": target_prompt_tokens,
        "actual_prompt_tokens": actual_prompt_tokens,
        "prompt_token_debug": prompt_token_debug,
        "generation": generation,
        "status": _completion_status(completion),
        "latency_ms": completion["latency_ms"],
        "ttft_ms": completion["ttft_ms"],
        "ttft_ms_effective": effective_ttft_ms,
        "ttft_source": "client_stream"
        if completion["ttft_ms"] is not None
        else ("server_metrics" if effective_ttft_ms is not None else None),
        "usage": completion["usage"],
        "finish_reason": completion["finish_reason"],
        "content_chars": len(completion["content"]),
        "reasoning_chars": len(completion["reasoning"]),
        "tool_call_count": len(completion["tool_calls"]),
        "raw_event_count": len(completion["raw_events"]),
        "data_event_count": completion["data_event_count"],
        "content_event_count": completion["content_event_count"],
        "reasoning_event_count": completion["reasoning_event_count"],
        "tool_call_event_count": completion["tool_call_event_count"],
        "end_to_end_tokens_per_second": completion["end_to_end_tokens_per_second"],
        "itl_stats": _itl_stats(completion.get("output_event_offsets_ms") or []),
        "server_stats": server_stats,
        "server_metrics_delta": metric_values,
        "telemetry_summary": telemetry_summary,
        "telemetry_path": telemetry_path,
        "telemetry_timeseries_path": telemetry_timeseries_path,
        "joules_per_request": joules_per_request,
        "joules_per_output_token": joules_per_output_token,
        "reasoning_only_truncated": completion["reasoning_only_truncated"],
        "empty_stream_bug": completion["empty_stream_bug"],
        "artifact_paths": artifact_paths,
        "prompt_token_debug_path": artifact_paths.get("prompt_token_debug_path"),
        "response_preview": completion["content"][:400],
    }
    if extra:
        record.update(extra)
    return record


def _execute_single_request(
    *,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment_id: str,
    experiment_kind: str,
    phase: str | None,
    response_base: Path,
    messages: list[dict[str, Any]],
    target_prompt_tokens: int,
    actual_prompt_tokens: int | None,
    prompt_token_debug: dict[str, Any] | None,
    generation: dict[str, Any],
    thinking_enabled: bool,
    seed: int | None,
    request_label: str | None = None,
    case_id: str | None = None,
    batch_id: str | None = None,
    batch_concurrency: int | None = None,
    with_tegrastats: bool = False,
    tegrastats_interval_ms: int = 500,
    capture_metrics: bool = True,
) -> dict[str, Any]:
    telemetry_path = response_base / "tegrastats.log"
    tegra = TegraStatsSession(telemetry_path, interval_ms=tegrastats_interval_ms)
    tegra_active = False
    if with_tegrastats:
        tegra_active = tegra.start()

    metrics_before = scrape_metrics(base_url, api_key) if capture_metrics else None
    try:
        completion = request_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=backend_fields["model"],
            messages=messages,
            generation=generation,
            thinking_enabled=thinking_enabled,
            prompt_token_estimate=actual_prompt_tokens,
            seed=seed,
        )
    finally:
        if tegra_active:
            tegra.stop()

    metrics_after = scrape_metrics(base_url, api_key) if capture_metrics else None
    metric_values = metrics_delta(metrics_before, metrics_after) if capture_metrics else None
    telemetry_summary = summarize_tegrastats_log(telemetry_path) if tegra_active else None
    telemetry_timeseries_path = (
        write_tegrastats_timeseries(telemetry_path, tegrastats_interval_ms)
        if tegra_active
        else None
    )
    artifact_paths = _write_completion_artifacts(response_base, completion, prompt_token_debug)
    return _build_request_record(
        backend_fields=backend_fields,
        experiment_id=experiment_id,
        experiment_kind=experiment_kind,
        phase=phase,
        response_id=response_base.name,
        request_label=request_label,
        target_prompt_tokens=target_prompt_tokens,
        actual_prompt_tokens=actual_prompt_tokens,
        prompt_token_debug=prompt_token_debug,
        generation=generation,
        thinking_enabled=thinking_enabled,
        seed=seed,
        completion=completion,
        metric_values=metric_values,
        telemetry_summary=telemetry_summary,
        telemetry_path=str(telemetry_path) if tegra_active else None,
        telemetry_timeseries_path=telemetry_timeseries_path,
        artifact_paths=artifact_paths,
        batch_id=batch_id,
        batch_concurrency=batch_concurrency,
        case_id=case_id,
    )


def _build_generation(
    max_tokens: int, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 64
) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def _batch_summary_record(
    *,
    backend_fields: dict[str, Any],
    experiment_id: str,
    experiment_kind: str,
    phase: str | None,
    case_id: str | None,
    batch_id: str,
    concurrency: int,
    batch_wall_time_ms: float,
    request_records: list[dict[str, Any]],
    metric_values: dict[str, Any] | None,
    telemetry_summary: dict[str, Any] | None,
    telemetry_path: str | None,
    telemetry_timeseries_path: str | None,
) -> dict[str, Any]:
    latencies = [
        record["latency_ms"]
        for record in request_records
        if isinstance(record.get("latency_ms"), (int, float))
    ]
    ttfts = [
        record["ttft_ms_effective"]
        for record in request_records
        if isinstance(record.get("ttft_ms_effective"), (int, float))
    ]
    completion_tokens_total = sum(
        (record.get("usage") or {}).get("completion_tokens") or 0 for record in request_records
    )
    server_ttft_ms = _metric_average(
        metric_values,
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
    )
    server_queue_ms = _metric_average(
        metric_values,
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
    )
    server_prefill_ms = _metric_average(
        metric_values,
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
    )
    server_decode_ms = _metric_average(
        metric_values,
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
    )
    throughput_tps = None
    if batch_wall_time_ms > 0:
        throughput_tps = round(completion_tokens_total / (batch_wall_time_ms / 1000.0), 3)
    joules_total = _compute_joules(telemetry_summary, batch_wall_time_ms)
    joules_per_request = None
    joules_per_output_token = None
    if joules_total is not None and request_records:
        joules_per_request = round(joules_total / len(request_records), 6)
    if joules_total is not None and completion_tokens_total:
        joules_per_output_token = round(joules_total / completion_tokens_total, 6)
    if server_ttft_ms is not None:
        server_ttft_ms = round(server_ttft_ms * 1000.0, 3)
    if server_queue_ms is not None:
        server_queue_ms = round(server_queue_ms * 1000.0, 3)
    if server_prefill_ms is not None:
        server_prefill_ms = round(server_prefill_ms * 1000.0, 3)
    if server_decode_ms is not None:
        server_decode_ms = round(server_decode_ms * 1000.0, 3)
    return {
        **backend_fields,
        "record_type": "batch",
        "experiment_id": experiment_id,
        "experiment_kind": experiment_kind,
        "phase": phase,
        "case_id": case_id,
        "batch_id": batch_id,
        "concurrency": concurrency,
        "request_count": len(request_records),
        "batch_wall_time_ms": round(batch_wall_time_ms, 3),
        "status": "completed",
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_p99_ms": _percentile(latencies, 0.99),
        "ttft_p50_ms": _percentile(ttfts, 0.50),
        "ttft_p95_ms": _percentile(ttfts, 0.95),
        "ttft_p99_ms": _percentile(ttfts, 0.99),
        "throughput_tokens_per_second": throughput_tps,
        "completion_tokens_total": completion_tokens_total,
        "server_ttft_ms": server_ttft_ms,
        "server_queue_ms": server_queue_ms,
        "server_prefill_ms": server_prefill_ms,
        "server_decode_ms": server_decode_ms,
        "server_metrics_delta": metric_values,
        "telemetry_summary": telemetry_summary,
        "telemetry_path": telemetry_path,
        "telemetry_timeseries_path": telemetry_timeseries_path,
        "joules_total": joules_total,
        "joules_per_request": joules_per_request,
        "joules_per_output_token": joules_per_output_token,
    }


def _run_thermal_soak(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    experiment_id = experiment["id"]
    duration_sec = int(params["duration_sec"])
    request_gap_sec = float(params.get("request_gap_sec", 0.0))
    target_prompt_tokens = int(params["target_prompt_tokens"])
    max_tokens = int(params["max_tokens"])
    thinking_enabled = bool(params.get("thinking_enabled", False))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 1000))
    generation = _build_generation(
        max_tokens=max_tokens,
        temperature=float(params.get("temperature", 0.7)),
        top_p=float(params.get("top_p", 0.95)),
        top_k=int(params.get("top_k", 64)),
    )

    messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
        base_url=base_url,
        api_key=api_key,
        model=backend_fields["model"],
        target_prompt_tokens=target_prompt_tokens,
        question="Identify the most likely performance issue described in the dossier and provide the minimum safe next step.",
        thinking_enabled=thinking_enabled,
    )

    experiment_response_dir = run_paths.responses_dir / experiment_id
    experiment_response_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = run_paths.telemetry_dir / f"{experiment_id}.log"
    tegra = TegraStatsSession(telemetry_path, interval_ms=tegrastats_interval_ms)
    tegra_active = False
    if with_tegrastats:
        tegra_active = tegra.start()

    request_index = 0
    start = time.perf_counter()
    try:
        while (time.perf_counter() - start) < duration_sec:
            request_index += 1
            response_base = experiment_response_dir / f"request_{request_index:04d}"
            record = _execute_single_request(
                base_url=base_url,
                api_key=api_key,
                backend_fields=backend_fields,
                experiment_id=experiment_id,
                experiment_kind=experiment["kind"],
                phase="thermal_iteration",
                response_base=response_base,
                messages=messages,
                target_prompt_tokens=target_prompt_tokens,
                actual_prompt_tokens=actual_prompt_tokens,
                prompt_token_debug=prompt_token_debug,
                generation=generation,
                thinking_enabled=thinking_enabled,
                seed=None if seed is None else seed + request_index,
                request_label=f"thermal-{request_index:04d}",
                with_tegrastats=False,
                capture_metrics=True,
            )
            record["elapsed_since_start_s"] = round(time.perf_counter() - start, 3)
            write_record(run_paths.records_path, record)
            if request_gap_sec > 0:
                time.sleep(request_gap_sec)
    finally:
        if tegra_active:
            tegra.stop()

    telemetry_summary = summarize_tegrastats_log(telemetry_path) if tegra_active else None
    telemetry_timeseries_path = (
        write_tegrastats_timeseries(telemetry_path, tegrastats_interval_ms)
        if tegra_active
        else None
    )
    write_record(
        run_paths.records_path,
        {
            **backend_fields,
            "record_type": "experiment_summary",
            "experiment_id": experiment_id,
            "experiment_kind": experiment["kind"],
            "duration_sec": duration_sec,
            "request_count": request_index,
            "telemetry_summary": telemetry_summary,
            "telemetry_path": str(telemetry_path) if tegra_active else None,
            "telemetry_timeseries_path": telemetry_timeseries_path,
        },
    )


def _run_concurrency_scaling(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    experiment_id = experiment["id"]
    concurrency_levels = [int(level) for level in params["concurrency_levels"]]
    rounds_per_level = int(params.get("rounds_per_level", 3))
    target_prompt_tokens = int(params["target_prompt_tokens"])
    max_tokens = int(params["max_tokens"])
    thinking_enabled = bool(params.get("thinking_enabled", False))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 500))
    generation = _build_generation(max_tokens=max_tokens)

    for level in concurrency_levels:
        for round_index in range(1, rounds_per_level + 1):
            batch_id = f"concurrency_{level:02d}_round_{round_index:02d}"
            prepared_requests = []
            for request_index in range(1, level + 1):
                request_label = f"c{level}_r{round_index}_q{request_index}"
                messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
                    base_url=base_url,
                    api_key=api_key,
                    model=backend_fields["model"],
                    target_prompt_tokens=target_prompt_tokens,
                    question="Summarize the likely root cause and list the next two actions.",
                    thinking_enabled=thinking_enabled,
                    request_label=request_label,
                    preserve_prefix=False,
                )
                prepared_requests.append(
                    {
                        "request_index": request_index,
                        "request_label": request_label,
                        "messages": messages,
                        "actual_prompt_tokens": actual_prompt_tokens,
                        "prompt_token_debug": prompt_token_debug,
                    }
                )
            telemetry_path = run_paths.telemetry_dir / f"{experiment_id}__{batch_id}.log"
            tegra = TegraStatsSession(telemetry_path, interval_ms=tegrastats_interval_ms)
            tegra_active = False
            if with_tegrastats:
                tegra_active = tegra.start()
            metrics_before = scrape_metrics(base_url, api_key)
            batch_start = time.perf_counter()
            futures = []
            request_records: list[dict[str, Any]] = []
            try:
                with ThreadPoolExecutor(max_workers=level) as executor:
                    for prepared in prepared_requests:
                        request_index = prepared["request_index"]
                        request_label = prepared["request_label"]
                        response_base = (
                            run_paths.responses_dir
                            / experiment_id
                            / batch_id
                            / f"request_{request_index:02d}"
                        )
                        request_seed = (
                            None
                            if seed is None
                            else seed + (level * 1000) + (round_index * 100) + request_index
                        )
                        futures.append(
                            executor.submit(
                                _execute_single_request,
                                base_url=base_url,
                                api_key=api_key,
                                backend_fields=backend_fields,
                                experiment_id=experiment_id,
                                experiment_kind=experiment["kind"],
                                phase="concurrency_request",
                                response_base=response_base,
                                messages=prepared["messages"],
                                target_prompt_tokens=target_prompt_tokens,
                                actual_prompt_tokens=prepared["actual_prompt_tokens"],
                                prompt_token_debug=prepared["prompt_token_debug"],
                                generation=generation,
                                thinking_enabled=thinking_enabled,
                                seed=request_seed,
                                request_label=request_label,
                                batch_id=batch_id,
                                batch_concurrency=level,
                                with_tegrastats=False,
                                capture_metrics=False,
                            )
                        )
                    for future in as_completed(futures):
                        request_records.append(future.result())
            finally:
                if tegra_active:
                    tegra.stop()
            batch_wall_time_ms = (time.perf_counter() - batch_start) * 1000.0
            metrics_after = scrape_metrics(base_url, api_key)
            metric_values = metrics_delta(metrics_before, metrics_after)
            telemetry_summary = summarize_tegrastats_log(telemetry_path) if tegra_active else None
            telemetry_timeseries_path = (
                write_tegrastats_timeseries(telemetry_path, tegrastats_interval_ms)
                if tegra_active
                else None
            )
            for record in sorted(request_records, key=lambda item: item["request_label"] or ""):
                write_record(run_paths.records_path, record)
            write_record(
                run_paths.records_path,
                _batch_summary_record(
                    backend_fields=backend_fields,
                    experiment_id=experiment_id,
                    experiment_kind=experiment["kind"],
                    phase="concurrency_batch",
                    case_id=None,
                    batch_id=batch_id,
                    concurrency=level,
                    batch_wall_time_ms=batch_wall_time_ms,
                    request_records=request_records,
                    metric_values=metric_values,
                    telemetry_summary=telemetry_summary,
                    telemetry_path=str(telemetry_path) if tegra_active else None,
                    telemetry_timeseries_path=telemetry_timeseries_path,
                ),
            )


def _run_io_length_sweep(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    repeats = int(params.get("repeats", 3))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 500))
    for case in params["cases"]:
        case_id = case["id"]
        target_prompt_tokens = int(case["target_prompt_tokens"])
        max_tokens = int(case["max_tokens"])
        thinking_enabled = bool(case.get("thinking_enabled", False))
        generation = _build_generation(max_tokens=max_tokens)
        for repeat_index in range(1, repeats + 1):
            messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
                base_url=base_url,
                api_key=api_key,
                model=backend_fields["model"],
                target_prompt_tokens=target_prompt_tokens,
                question=case["question"],
                thinking_enabled=thinking_enabled,
                request_label=f"{case_id}-{repeat_index:02d}",
                preserve_prefix=False,
            )
            response_base = (
                run_paths.responses_dir / experiment["id"] / case_id / f"repeat_{repeat_index:02d}"
            )
            record = _execute_single_request(
                base_url=base_url,
                api_key=api_key,
                backend_fields=backend_fields,
                experiment_id=experiment["id"],
                experiment_kind=experiment["kind"],
                phase="io_length_case",
                response_base=response_base,
                messages=messages,
                target_prompt_tokens=target_prompt_tokens,
                actual_prompt_tokens=actual_prompt_tokens,
                prompt_token_debug=prompt_token_debug,
                generation=generation,
                thinking_enabled=thinking_enabled,
                seed=None if seed is None else seed + repeat_index,
                request_label=f"{case_id}-{repeat_index:02d}",
                case_id=case_id,
                with_tegrastats=with_tegrastats,
                tegrastats_interval_ms=tegrastats_interval_ms,
                capture_metrics=True,
            )
            write_record(run_paths.records_path, record)


def _run_itl_distribution(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    repeats = int(params.get("repeats", 8))
    target_prompt_tokens = int(params["target_prompt_tokens"])
    max_tokens = int(params["max_tokens"])
    thinking_enabled = bool(params.get("thinking_enabled", False))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 500))
    generation = _build_generation(max_tokens=max_tokens)
    for repeat_index in range(1, repeats + 1):
        messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
            base_url=base_url,
            api_key=api_key,
            model=backend_fields["model"],
            target_prompt_tokens=target_prompt_tokens,
            question="Stream the answer smoothly and list the main performance signals in descending order of importance.",
            thinking_enabled=thinking_enabled,
            request_label=f"itl-{repeat_index:02d}",
        )
        response_base = run_paths.responses_dir / experiment["id"] / f"repeat_{repeat_index:02d}"
        record = _execute_single_request(
            base_url=base_url,
            api_key=api_key,
            backend_fields=backend_fields,
            experiment_id=experiment["id"],
            experiment_kind=experiment["kind"],
            phase="itl_request",
            response_base=response_base,
            messages=messages,
            target_prompt_tokens=target_prompt_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
            prompt_token_debug=prompt_token_debug,
            generation=generation,
            thinking_enabled=thinking_enabled,
            seed=None if seed is None else seed + repeat_index,
            request_label=f"itl-{repeat_index:02d}",
            with_tegrastats=with_tegrastats,
            tegrastats_interval_ms=tegrastats_interval_ms,
            capture_metrics=True,
        )
        write_record(run_paths.records_path, record)


def _run_energy_efficiency(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    repeats = int(params.get("repeats", 3))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 200))
    for case in params["cases"]:
        case_id = case["id"]
        target_prompt_tokens = int(case["target_prompt_tokens"])
        max_tokens = int(case["max_tokens"])
        thinking_enabled = bool(case.get("thinking_enabled", False))
        generation = _build_generation(max_tokens=max_tokens)
        for repeat_index in range(1, repeats + 1):
            messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
                base_url=base_url,
                api_key=api_key,
                model=backend_fields["model"],
                target_prompt_tokens=target_prompt_tokens,
                question=case["question"],
                thinking_enabled=thinking_enabled,
                request_label=f"{case_id}-{repeat_index:02d}",
            )
            response_base = (
                run_paths.responses_dir / experiment["id"] / case_id / f"repeat_{repeat_index:02d}"
            )
            record = _execute_single_request(
                base_url=base_url,
                api_key=api_key,
                backend_fields=backend_fields,
                experiment_id=experiment["id"],
                experiment_kind=experiment["kind"],
                phase="energy_request",
                response_base=response_base,
                messages=messages,
                target_prompt_tokens=target_prompt_tokens,
                actual_prompt_tokens=actual_prompt_tokens,
                prompt_token_debug=prompt_token_debug,
                generation=generation,
                thinking_enabled=thinking_enabled,
                seed=None if seed is None else seed + repeat_index,
                request_label=f"{case_id}-{repeat_index:02d}",
                case_id=case_id,
                with_tegrastats=True,
                tegrastats_interval_ms=tegrastats_interval_ms,
                capture_metrics=True,
            )
            write_record(run_paths.records_path, record)


def _run_kv_cache_saturation(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    concurrency_levels = [int(level) for level in params["concurrency_levels"]]
    rounds_per_level = int(params.get("rounds_per_level", 2))
    target_prompt_tokens = int(params["target_prompt_tokens"])
    max_tokens = int(params["max_tokens"])
    thinking_enabled = bool(params.get("thinking_enabled", False))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 500))
    generation = _build_generation(max_tokens=max_tokens)

    for level in concurrency_levels:
        for round_index in range(1, rounds_per_level + 1):
            batch_id = f"kv_{level:02d}_round_{round_index:02d}"
            prepared_requests = []
            for request_index in range(1, level + 1):
                request_label = f"kv{level}_r{round_index}_q{request_index}"
                messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
                    base_url=base_url,
                    api_key=api_key,
                    model=backend_fields["model"],
                    target_prompt_tokens=target_prompt_tokens,
                    question="Synthesize the dossier into a concise root cause explanation and a three-step action plan.",
                    thinking_enabled=thinking_enabled,
                    request_label=request_label,
                )
                prepared_requests.append(
                    {
                        "request_index": request_index,
                        "request_label": request_label,
                        "messages": messages,
                        "actual_prompt_tokens": actual_prompt_tokens,
                        "prompt_token_debug": prompt_token_debug,
                    }
                )
            telemetry_path = run_paths.telemetry_dir / f"{experiment['id']}__{batch_id}.log"
            tegra = TegraStatsSession(telemetry_path, interval_ms=tegrastats_interval_ms)
            tegra_active = False
            if with_tegrastats:
                tegra_active = tegra.start()
            metrics_before = scrape_metrics(base_url, api_key)
            batch_start = time.perf_counter()
            futures = []
            request_records: list[dict[str, Any]] = []
            try:
                with ThreadPoolExecutor(max_workers=level) as executor:
                    for prepared in prepared_requests:
                        request_index = prepared["request_index"]
                        request_label = prepared["request_label"]
                        response_base = (
                            run_paths.responses_dir
                            / experiment["id"]
                            / batch_id
                            / f"request_{request_index:02d}"
                        )
                        futures.append(
                            executor.submit(
                                _execute_single_request,
                                base_url=base_url,
                                api_key=api_key,
                                backend_fields=backend_fields,
                                experiment_id=experiment["id"],
                                experiment_kind=experiment["kind"],
                                phase="kv_pressure_request",
                                response_base=response_base,
                                messages=prepared["messages"],
                                target_prompt_tokens=target_prompt_tokens,
                                actual_prompt_tokens=prepared["actual_prompt_tokens"],
                                prompt_token_debug=prepared["prompt_token_debug"],
                                generation=generation,
                                thinking_enabled=thinking_enabled,
                                seed=None
                                if seed is None
                                else seed + (level * 1000) + request_index,
                                request_label=request_label,
                                batch_id=batch_id,
                                batch_concurrency=level,
                                with_tegrastats=False,
                                capture_metrics=False,
                            )
                        )
                    for future in as_completed(futures):
                        request_records.append(future.result())
            finally:
                if tegra_active:
                    tegra.stop()
            batch_wall_time_ms = (time.perf_counter() - batch_start) * 1000.0
            metrics_after = scrape_metrics(base_url, api_key)
            metric_values = metrics_delta(metrics_before, metrics_after)
            telemetry_summary = summarize_tegrastats_log(telemetry_path) if tegra_active else None
            telemetry_timeseries_path = (
                write_tegrastats_timeseries(telemetry_path, tegrastats_interval_ms)
                if tegra_active
                else None
            )
            for record in sorted(request_records, key=lambda item: item["request_label"] or ""):
                write_record(run_paths.records_path, record)
            write_record(
                run_paths.records_path,
                _batch_summary_record(
                    backend_fields=backend_fields,
                    experiment_id=experiment["id"],
                    experiment_kind=experiment["kind"],
                    phase="kv_pressure_batch",
                    case_id=None,
                    batch_id=batch_id,
                    concurrency=level,
                    batch_wall_time_ms=batch_wall_time_ms,
                    request_records=request_records,
                    metric_values=metric_values,
                    telemetry_summary=telemetry_summary,
                    telemetry_path=str(telemetry_path) if tegra_active else None,
                    telemetry_timeseries_path=telemetry_timeseries_path,
                ),
            )


def _run_cold_start(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    target_prompt_tokens = int(params["target_prompt_tokens"])
    max_tokens = int(params["max_tokens"])
    thinking_enabled = bool(params.get("thinking_enabled", False))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 500))
    generation = _build_generation(max_tokens=max_tokens)
    idle_windows = [0] + [int(value) for value in params.get("idle_windows_sec", [])]
    for phase_index, idle_sec in enumerate(idle_windows, start=1):
        phase_name = "post_startup_first_request" if phase_index == 1 else f"idle_{idle_sec}s"
        if phase_index > 1 and idle_sec > 0:
            time.sleep(idle_sec)
        messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
            base_url=base_url,
            api_key=api_key,
            model=backend_fields["model"],
            target_prompt_tokens=target_prompt_tokens,
            question="Answer with a short diagnosis and one next action.",
            thinking_enabled=thinking_enabled,
            request_label=phase_name,
        )
        response_base = run_paths.responses_dir / experiment["id"] / phase_name
        record = _execute_single_request(
            base_url=base_url,
            api_key=api_key,
            backend_fields=backend_fields,
            experiment_id=experiment["id"],
            experiment_kind=experiment["kind"],
            phase=phase_name,
            response_base=response_base,
            messages=messages,
            target_prompt_tokens=target_prompt_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
            prompt_token_debug=prompt_token_debug,
            generation=generation,
            thinking_enabled=thinking_enabled,
            seed=None if seed is None else seed + phase_index,
            request_label=phase_name,
            with_tegrastats=with_tegrastats,
            tegrastats_interval_ms=tegrastats_interval_ms,
            capture_metrics=True,
        )
        record["idle_before_request_sec"] = idle_sec
        write_record(run_paths.records_path, record)


def _run_prefix_caching(
    *,
    run_paths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    experiment: dict[str, Any],
    seed: int | None,
    with_tegrastats: bool,
) -> None:
    params = experiment["params"]
    target_prompt_tokens = int(params["target_prompt_tokens"])
    max_tokens = int(params["max_tokens"])
    thinking_enabled = bool(params.get("thinking_enabled", False))
    tegrastats_interval_ms = int(params.get("tegrastats_interval_ms", 500))
    generation = _build_generation(max_tokens=max_tokens)
    questions = list(params["questions"])
    for request_index, question in enumerate(questions, start=1):
        request_label = f"prefix_{request_index:02d}"
        messages, actual_prompt_tokens, prompt_token_debug = _build_synthetic_messages(
            base_url=base_url,
            api_key=api_key,
            model=backend_fields["model"],
            target_prompt_tokens=target_prompt_tokens,
            question=question,
            thinking_enabled=thinking_enabled,
            request_label=request_label,
            preserve_prefix=True,
        )
        response_base = run_paths.responses_dir / experiment["id"] / request_label
        record = _execute_single_request(
            base_url=base_url,
            api_key=api_key,
            backend_fields=backend_fields,
            experiment_id=experiment["id"],
            experiment_kind=experiment["kind"],
            phase="prefix_request",
            response_base=response_base,
            messages=messages,
            target_prompt_tokens=target_prompt_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
            prompt_token_debug=prompt_token_debug,
            generation=generation,
            thinking_enabled=thinking_enabled,
            seed=None if seed is None else seed + request_index,
            request_label=request_label,
            with_tegrastats=with_tegrastats,
            tegrastats_interval_ms=tegrastats_interval_ms,
            capture_metrics=True,
        )
        record["question"] = question
        record["request_index"] = request_index
        write_record(run_paths.records_path, record)


EXPERIMENT_RUNNERS = {
    "thermal_soak": _run_thermal_soak,
    "concurrency_scaling": _run_concurrency_scaling,
    "io_length_sweep": _run_io_length_sweep,
    "itl_distribution": _run_itl_distribution,
    "energy_efficiency": _run_energy_efficiency,
    "kv_cache_saturation": _run_kv_cache_saturation,
    "cold_start": _run_cold_start,
    "prefix_caching": _run_prefix_caching,
}


def run_systems_suite(
    *,
    project_root: Path,
    backend_config_path: Path,
    output_root: Path,
    manifest_path: Path | None = None,
    experiments: set[str] | None = None,
    dry_run: bool = False,
    with_tegrastats: bool = False,
    seed: int | None = 20260417,
) -> Path:
    backend = load_yaml(backend_config_path)
    backend_fields = _backend_record_fields(backend, backend_config_path)
    effective_manifest_path = manifest_path or (project_root / "systems" / "manifest.yaml")
    manifest = load_yaml(effective_manifest_path)
    run_paths = ensure_run_dirs(output_root, dry_run=dry_run)

    base_url = backend["base_url"]
    api_key = backend.get("api_key")
    model = backend.get("model")
    if not model:
        raise ValueError(f"Backend config {backend_config_path} must pin an explicit model id.")

    model_verification = verify_model(base_url, api_key, model)
    if model_verification["advertised_models"] and not model_verification["requested_model_found"]:
        raise ValueError(
            f"Requested model {model!r} was not advertised by {base_url}. "
            f"Advertised models: {model_verification['advertised_models']}"
        )

    collect_run_metadata(
        run_paths=run_paths,
        backend=backend,
        model_verification=model_verification,
        run_options={
            "suite_type": "systems",
            "dry_run": dry_run,
            "with_tegrastats": with_tegrastats,
            "seed": seed,
            "experiments": sorted(experiments) if experiments else None,
            "systems_manifest_path": str(effective_manifest_path),
            "api_mode": "chat_completions",
            "token_estimation_mode": "chat_tokenize_endpoint",
        },
    )

    selected = []
    for experiment in manifest["experiments"]:
        if experiments and experiment["id"] not in experiments:
            continue
        selected.append(experiment)

    if dry_run:
        for experiment in selected:
            write_record(
                run_paths.records_path,
                {
                    **backend_fields,
                    "record_type": "plan",
                    "experiment_id": experiment["id"],
                    "experiment_kind": experiment["kind"],
                    "title": experiment["title"],
                    "description": experiment["description"],
                    "params": experiment["params"],
                    "dry_run": True,
                },
            )
        return run_paths.root

    experiment_results: list[dict[str, Any]] = []
    for experiment_index, experiment in enumerate(selected, start=1):
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            runner = EXPERIMENT_RUNNERS[experiment["kind"]]
            runner(
                run_paths=run_paths,
                base_url=base_url,
                api_key=api_key,
                backend_fields=backend_fields,
                experiment=experiment,
                seed=None if seed is None else seed + experiment_index * 10000,
                with_tegrastats=with_tegrastats,
            )
            experiment_results.append(
                {
                    "experiment_id": experiment["id"],
                    "status": "completed",
                    "started_at": started_at,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
        except Exception as exc:
            write_record(
                run_paths.records_path,
                {
                    **backend_fields,
                    "record_type": "experiment_error",
                    "experiment_id": experiment["id"],
                    "experiment_kind": experiment["kind"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            experiment_results.append(
                {
                    "experiment_id": experiment["id"],
                    "status": "error",
                    "started_at": started_at,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    update_run_metadata(
        run_paths,
        lambda metadata: metadata.update(
            {
                "systems_manifest_path": str(effective_manifest_path),
                "experiment_results": experiment_results,
                "experiment_error_count": sum(
                    1 for item in experiment_results if item["status"] == "error"
                ),
            }
        ),
    )
    return run_paths.root
