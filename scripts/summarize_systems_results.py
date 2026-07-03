#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _extract_usage_value(record: dict[str, Any], key: str) -> Any:
    usage = record.get("usage") or {}
    return usage.get(key)


def _row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    server_stats = record.get("server_stats") or {}
    telemetry = record.get("telemetry_summary") or {}
    itl = record.get("itl_stats") or {}
    return {
        "record_type": record.get("record_type"),
        "experiment_id": record.get("experiment_id"),
        "experiment_kind": record.get("experiment_kind"),
        "backend": record.get("backend"),
        "backend_profile": record.get("backend_profile"),
        "backend_config_path": record.get("backend_config_path"),
        "model": record.get("model"),
        "configured_max_context_tokens": record.get("configured_max_context_tokens"),
        "max_soft_tokens": record.get("max_soft_tokens"),
        "vision_budget_label": record.get("vision_budget_label"),
        "prefix_caching_enabled": record.get("prefix_caching_enabled"),
        "supported_modalities": ",".join(record.get("supported_modalities") or []),
        "audio_supported": record.get("audio_supported"),
        "container_image": record.get("container_image"),
        "container_image_tag": record.get("container_image_tag"),
        "container_image_digest": record.get("container_image_digest"),
        "phase": record.get("phase"),
        "case_id": record.get("case_id"),
        "batch_id": record.get("batch_id"),
        "batch_concurrency": record.get("batch_concurrency") or record.get("concurrency"),
        "request_label": record.get("request_label"),
        "response_id": record.get("response_id"),
        "status": record.get("status"),
        "thinking_enabled": record.get("thinking_enabled"),
        "seed": record.get("seed"),
        "target_prompt_tokens": record.get("target_prompt_tokens"),
        "actual_prompt_tokens": record.get("actual_prompt_tokens"),
        "prompt_token_debug_ok": (record.get("prompt_token_debug") or {}).get("ok"),
        "prompt_token_debug_status_code": (record.get("prompt_token_debug") or {}).get(
            "status_code"
        ),
        "prompt_token_debug_count": (record.get("prompt_token_debug") or {}).get("count"),
        "prompt_token_debug_error_type": (record.get("prompt_token_debug") or {}).get("error_type"),
        "prompt_token_debug_path": record.get("prompt_token_debug_path")
        or ((record.get("artifact_paths") or {}).get("prompt_token_debug_path")),
        "max_tokens": (record.get("generation") or {}).get("max_tokens"),
        "latency_ms": record.get("latency_ms"),
        "ttft_ms": record.get("ttft_ms"),
        "ttft_ms_effective": record.get("ttft_ms_effective"),
        "end_to_end_tokens_per_second": record.get("end_to_end_tokens_per_second"),
        "prompt_tokens": _extract_usage_value(record, "prompt_tokens"),
        "completion_tokens": _extract_usage_value(record, "completion_tokens"),
        "finish_reason": record.get("finish_reason"),
        "content_chars": record.get("content_chars"),
        "reasoning_chars": record.get("reasoning_chars"),
        "tool_call_count": record.get("tool_call_count"),
        "raw_event_count": record.get("raw_event_count"),
        "data_event_count": record.get("data_event_count"),
        "content_event_count": record.get("content_event_count"),
        "reasoning_event_count": record.get("reasoning_event_count"),
        "tool_call_event_count": record.get("tool_call_event_count"),
        "reasoning_only_truncated": (record.get("reasoning_only_truncated") or {}).get("detected"),
        "empty_stream_bug": (record.get("empty_stream_bug") or {}).get("detected"),
        "itl_p50_ms": itl.get("itl_p50_ms"),
        "itl_p95_ms": itl.get("itl_p95_ms"),
        "itl_p99_ms": itl.get("itl_p99_ms"),
        "itl_max_ms": itl.get("itl_max_ms"),
        "stall_count": itl.get("stall_count"),
        "server_ttft_ms": server_stats.get("server_ttft_ms"),
        "server_e2e_latency_ms": server_stats.get("server_e2e_latency_ms"),
        "server_prefill_ms": server_stats.get("server_prefill_ms"),
        "server_decode_ms": server_stats.get("server_decode_ms"),
        "server_queue_ms": server_stats.get("server_queue_ms"),
        "server_decode_tokens_per_second": server_stats.get("server_decode_tokens_per_second"),
        "server_kv_cache_usage_perc": server_stats.get("server_kv_cache_usage_perc"),
        "batch_wall_time_ms": record.get("batch_wall_time_ms"),
        "latency_p50_ms": record.get("latency_p50_ms"),
        "latency_p95_ms": record.get("latency_p95_ms"),
        "latency_p99_ms": record.get("latency_p99_ms"),
        "ttft_p50_ms": record.get("ttft_p50_ms"),
        "ttft_p95_ms": record.get("ttft_p95_ms"),
        "ttft_p99_ms": record.get("ttft_p99_ms"),
        "throughput_tokens_per_second": record.get("throughput_tokens_per_second"),
        "completion_tokens_total": record.get("completion_tokens_total"),
        "joules_per_request": record.get("joules_per_request"),
        "joules_per_output_token": record.get("joules_per_output_token"),
        "joules_total": record.get("joules_total"),
        "telemetry_sample_count": telemetry.get("sample_count"),
        "telemetry_gpu_util_percent_avg": telemetry.get("gpu_util_percent_avg"),
        "telemetry_gpu_util_percent_peak": telemetry.get("gpu_util_percent_peak"),
        "telemetry_max_temp_c_avg": telemetry.get("max_temp_c_avg"),
        "telemetry_max_temp_c_peak": telemetry.get("max_temp_c_peak"),
        "telemetry_power_current_mw_avg": telemetry.get("power_current_mw_avg"),
        "telemetry_power_current_mw_peak": telemetry.get("power_current_mw_peak"),
        "telemetry_path": record.get("telemetry_path"),
        "telemetry_timeseries_path": record.get("telemetry_timeseries_path"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["record_type"],
            row["experiment_id"],
            row["experiment_kind"],
            row["backend"],
            row["backend_profile"],
            row["vision_budget_label"],
            row["max_soft_tokens"],
            row["prefix_caching_enabled"],
            row["phase"],
            row["case_id"],
            row["batch_concurrency"],
        )
        groups[key].append(row)

    numeric_fields = [
        "latency_ms",
        "ttft_ms_effective",
        "end_to_end_tokens_per_second",
        "completion_tokens",
        "itl_p50_ms",
        "itl_p95_ms",
        "itl_p99_ms",
        "server_ttft_ms",
        "server_prefill_ms",
        "server_decode_ms",
        "server_queue_ms",
        "server_decode_tokens_per_second",
        "server_kv_cache_usage_perc",
        "throughput_tokens_per_second",
        "completion_tokens_total",
        "joules_per_request",
        "joules_per_output_token",
        "telemetry_gpu_util_percent_avg",
        "telemetry_max_temp_c_peak",
        "telemetry_power_current_mw_avg",
    ]

    aggregate_rows = []
    for key, bucket in sorted(groups.items()):
        (
            record_type,
            experiment_id,
            experiment_kind,
            backend,
            backend_profile,
            vision_budget_label,
            max_soft_tokens,
            prefix_caching_enabled,
            phase,
            case_id,
            batch_concurrency,
        ) = key
        aggregate = {
            "record_type": record_type,
            "experiment_id": experiment_id,
            "experiment_kind": experiment_kind,
            "backend": backend,
            "backend_profile": backend_profile,
            "vision_budget_label": vision_budget_label,
            "max_soft_tokens": max_soft_tokens,
            "prefix_caching_enabled": prefix_caching_enabled,
            "phase": phase,
            "case_id": case_id,
            "batch_concurrency": batch_concurrency,
            "sample_count": len(bucket),
            "status_set": ",".join(sorted({str(row["status"]) for row in bucket})),
        }
        for field in numeric_fields:
            values = [row[field] for row in bucket if isinstance(row.get(field), (int, float))]
            if not values:
                continue
            aggregate[f"{field}_mean"] = round(statistics.mean(values), 3)
            aggregate[f"{field}_median"] = round(statistics.median(values), 3)
            aggregate[f"{field}_min"] = round(min(values), 3)
            aggregate[f"{field}_max"] = round(max(values), 3)
            aggregate[f"{field}_stdev"] = (
                round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(aggregate)
    return aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--include-dry-run", action="store_true")
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"Missing records file: {records_path}")

    rows = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("dry_run") and not args.include_dry_run:
                continue
            rows.append(_row_from_record(record))

    if not rows:
        raise SystemExit("No systems benchmark records found.")

    summary_path = args.run_dir / "systems_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = _aggregate(rows)
    aggregate_path = args.run_dir / "systems_condition_summary.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(summary_path)
    print(aggregate_path)


if __name__ == "__main__":
    main()
