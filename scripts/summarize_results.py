#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _usage_value(turn: dict[str, Any], key: str, fallback_key: str | None = None) -> Any:
    usage = turn.get("usage") or {}
    if usage.get(key) is not None:
        return usage.get(key)
    if fallback_key and usage.get(fallback_key) is not None:
        return usage.get(fallback_key)
    return None


def _row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    turns = record.get("turns", [])
    first_turn = turns[0] if turns else {}
    final_turn = turns[-1] if turns else {}
    first_stats = first_turn.get("server_stats") or {}
    final_stats = final_turn.get("server_stats") or {}
    telemetry = record.get("telemetry_summary") or {}
    return {
        "scenario_id": record.get("scenario_id"),
        "use_case_title": record.get("use_case_title"),
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
        "thinking_enabled": record.get("thinking_enabled"),
        "repeat_index": record.get("repeat_index"),
        "mode": record.get("mode"),
        "family": record.get("family"),
        "input_language": record.get("input_language"),
        "expected_output_language": record.get("expected_output_language"),
        "language_variant": record.get("language_variant"),
        "review_scope": record.get("review_scope"),
        "variant_id": record.get("variant_id"),
        "image_files": ",".join(record.get("image_files") or []),
        "scenario_connectivity": record.get("scenario_connectivity"),
        "execution_mode": record.get("execution_mode"),
        "status": record.get("status", "completed"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "dry_run": record.get("dry_run", False),
        "turn_count": len(turns),
        "seed": record.get("seed"),
        "generation_profile": record.get("generation_profile"),
        "max_context_tokens": record.get("max_context_tokens"),
        "scenario_declared_max_context_tokens": record.get("scenario_declared_max_context_tokens"),
        "context_budget_valid": record.get("context_budget_valid"),
        "prompt_truncated": record.get("prompt_truncated"),
        "original_prompt_token_estimate": record.get("original_prompt_token_estimate"),
        "truncated_context_files": ",".join(record.get("truncated_context_files") or []),
        "initial_prompt_token_estimate": record.get("initial_prompt_token_estimate"),
        "initial_prompt_token_debug_ok": (record.get("initial_prompt_token_debug") or {}).get("ok"),
        "initial_prompt_token_debug_status_code": (record.get("initial_prompt_token_debug") or {}).get("status_code"),
        "initial_prompt_token_debug_count": (record.get("initial_prompt_token_debug") or {}).get("count"),
        "initial_prompt_token_debug_error_type": (record.get("initial_prompt_token_debug") or {}).get("error_type"),
        "initial_prompt_token_debug_path": record.get("initial_prompt_token_debug_path"),
        "total_latency_ms": record.get("total_latency_ms"),
        "first_prompt_token_estimate": first_turn.get("prompt_token_estimate"),
        "first_prompt_token_debug_ok": (first_turn.get("prompt_token_debug") or {}).get("ok"),
        "first_prompt_token_debug_status_code": (first_turn.get("prompt_token_debug") or {}).get("status_code"),
        "first_prompt_token_debug_count": (first_turn.get("prompt_token_debug") or {}).get("count"),
        "first_prompt_token_debug_path": first_turn.get("prompt_token_debug_path"),
        "first_latency_ms": first_turn.get("latency_ms"),
        "first_ttft_ms": first_turn.get("ttft_ms"),
        "first_ttft_ms_effective": first_turn.get("ttft_ms_effective"),
        "first_ttft_source": first_turn.get("ttft_source"),
        "first_end_to_end_tokens_per_second": first_turn.get("end_to_end_tokens_per_second"),
        "first_finish_reason": first_turn.get("finish_reason"),
        "first_prompt_tokens": _usage_value(first_turn, "prompt_tokens", "prompt_tokens_server"),
        "first_completion_tokens": _usage_value(first_turn, "completion_tokens", "completion_tokens_server"),
        "first_completion_tokens_server_reported": (first_turn.get("usage") or {}).get("completion_tokens_server_reported"),
        "first_completion_tokens_source": (first_turn.get("usage") or {}).get("completion_tokens_source"),
        "first_raw_event_count": first_turn.get("raw_event_count"),
        "first_content_event_count": first_turn.get("content_event_count"),
        "first_reasoning_event_count": first_turn.get("reasoning_event_count"),
        "first_tool_call_event_count": first_turn.get("tool_call_event_count"),
        "first_data_event_count": first_turn.get("data_event_count"),
        "first_non_empty_text_event_count": first_turn.get("non_empty_text_event_count"),
        "first_reasoning_chars": len(first_turn.get("raw_reasoning") or ""),
        "first_raw_event_timeline_path": first_turn.get("raw_event_timeline_path"),
        "first_completion_token_fallback_count": ((first_turn.get("completion_token_fallback_debug") or {}).get("count")),
        "first_completion_token_fallback_debug_path": first_turn.get("completion_token_fallback_debug_path"),
        "first_reasoning_only_truncated": ((first_turn.get("reasoning_only_truncated") or {}).get("detected")),
        "first_empty_stream_bug": ((first_turn.get("empty_stream_bug") or {}).get("detected")),
        "first_server_decode_tokens_per_second": first_stats.get("server_decode_tokens_per_second"),
        "first_server_ttft_ms": first_stats.get("server_ttft_ms"),
        "first_server_e2e_latency_ms": first_stats.get("server_e2e_latency_ms"),
        "first_server_prefill_ms": first_stats.get("server_prefill_ms"),
        "first_server_decode_ms": first_stats.get("server_decode_ms"),
        "first_server_queue_ms": first_stats.get("server_queue_ms"),
        "first_server_kv_cache_usage": first_stats.get("server_kv_cache_usage_perc")
        or first_stats.get("server_kv_cache_usage_ratio"),
        "tool_call_count_total": sum(len(turn.get("parsed_tool_calls", [])) for turn in turns),
        "final_latency_ms": final_turn.get("latency_ms"),
        "final_prompt_token_estimate": final_turn.get("prompt_token_estimate"),
        "final_prompt_token_debug_ok": (final_turn.get("prompt_token_debug") or {}).get("ok"),
        "final_prompt_token_debug_status_code": (final_turn.get("prompt_token_debug") or {}).get("status_code"),
        "final_prompt_token_debug_count": (final_turn.get("prompt_token_debug") or {}).get("count"),
        "final_prompt_token_debug_path": final_turn.get("prompt_token_debug_path"),
        "final_ttft_ms": final_turn.get("ttft_ms"),
        "final_ttft_ms_effective": final_turn.get("ttft_ms_effective"),
        "final_ttft_source": final_turn.get("ttft_source"),
        "final_end_to_end_tokens_per_second": final_turn.get("end_to_end_tokens_per_second"),
        "final_prompt_tokens": _usage_value(final_turn, "prompt_tokens", "prompt_tokens_server"),
        "final_completion_tokens": _usage_value(final_turn, "completion_tokens", "completion_tokens_server"),
        "final_completion_tokens_server_reported": (final_turn.get("usage") or {}).get("completion_tokens_server_reported"),
        "final_completion_tokens_source": (final_turn.get("usage") or {}).get("completion_tokens_source"),
        "final_raw_event_count": final_turn.get("raw_event_count"),
        "final_content_event_count": final_turn.get("content_event_count"),
        "final_reasoning_event_count": final_turn.get("reasoning_event_count"),
        "final_tool_call_event_count": final_turn.get("tool_call_event_count"),
        "final_data_event_count": final_turn.get("data_event_count"),
        "final_non_empty_text_event_count": final_turn.get("non_empty_text_event_count"),
        "final_reasoning_chars": len(final_turn.get("raw_reasoning") or ""),
        "final_raw_event_timeline_path": final_turn.get("raw_event_timeline_path"),
        "final_completion_token_fallback_count": ((final_turn.get("completion_token_fallback_debug") or {}).get("count")),
        "final_completion_token_fallback_debug_path": final_turn.get("completion_token_fallback_debug_path"),
        "final_reasoning_only_truncated": ((final_turn.get("reasoning_only_truncated") or {}).get("detected")),
        "final_empty_stream_bug": ((final_turn.get("empty_stream_bug") or {}).get("detected")),
        "final_server_decode_tokens_per_second": final_stats.get("server_decode_tokens_per_second"),
        "final_server_ttft_ms": final_stats.get("server_ttft_ms"),
        "final_server_e2e_latency_ms": final_stats.get("server_e2e_latency_ms"),
        "final_server_prefill_ms": final_stats.get("server_prefill_ms"),
        "final_server_decode_ms": final_stats.get("server_decode_ms"),
        "final_server_queue_ms": final_stats.get("server_queue_ms"),
        "final_server_kv_cache_usage": final_stats.get("server_kv_cache_usage_perc")
        or final_stats.get("server_kv_cache_usage_ratio"),
        "telemetry_sample_count": telemetry.get("sample_count"),
        "telemetry_ram_used_mb_avg": telemetry.get("ram_used_mb_avg"),
        "telemetry_ram_used_mb_peak": telemetry.get("ram_used_mb_peak"),
        "telemetry_gpu_util_percent_avg": telemetry.get("gpu_util_percent_avg"),
        "telemetry_gpu_util_percent_peak": telemetry.get("gpu_util_percent_peak"),
        "telemetry_cpu_util_avg_percent_avg": telemetry.get("cpu_util_avg_percent_avg"),
        "telemetry_cpu_util_max_percent_peak": telemetry.get("cpu_util_max_percent_peak"),
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
            row["scenario_id"],
            row["backend"],
            row["backend_profile"],
            row["vision_budget_label"],
            row["max_soft_tokens"],
            row["prefix_caching_enabled"],
            row["model"],
            row["thinking_enabled"],
            row["family"],
            row["input_language"],
            row["expected_output_language"],
            row["language_variant"],
            row["mode"],
        )
        groups[key].append(row)

    numeric_fields = [
        "total_latency_ms",
        "first_ttft_ms",
        "first_ttft_ms_effective",
        "first_end_to_end_tokens_per_second",
        "first_server_decode_tokens_per_second",
        "final_ttft_ms",
        "final_ttft_ms_effective",
        "final_end_to_end_tokens_per_second",
        "final_server_decode_tokens_per_second",
        "final_completion_tokens",
        "telemetry_gpu_util_percent_avg",
        "telemetry_max_temp_c_peak",
        "telemetry_power_current_mw_avg",
    ]

    aggregate_rows = []
    for key, bucket in sorted(groups.items()):
        (
            scenario_id,
            backend,
            backend_profile,
            vision_budget_label,
            max_soft_tokens,
            prefix_caching_enabled,
            model,
            thinking_enabled,
            family,
            input_language,
            expected_output_language,
            language_variant,
            mode,
        ) = key
        aggregate = {
            "scenario_id": scenario_id,
            "backend": backend,
            "backend_profile": backend_profile,
            "vision_budget_label": vision_budget_label,
            "max_soft_tokens": max_soft_tokens,
            "prefix_caching_enabled": prefix_caching_enabled,
            "model": model,
            "thinking_enabled": thinking_enabled,
            "family": family,
            "input_language": input_language,
            "expected_output_language": expected_output_language,
            "language_variant": language_variant,
            "mode": mode,
            "max_context_tokens": bucket[0].get("max_context_tokens"),
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
            aggregate[f"{field}_stdev"] = round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
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
        raise SystemExit("No non-dry-run benchmark records found. Run a live benchmark or pass --include-dry-run.")

    output_path = args.run_dir / "summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = _aggregate(rows)
    aggregate_path = args.run_dir / "condition_summary.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(output_path)
    print(aggregate_path)


if __name__ == "__main__":
    main()
