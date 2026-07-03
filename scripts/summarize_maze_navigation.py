#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 3)


def _row_from_run(record: dict[str, Any]) -> dict[str, Any]:
    telemetry = record.get("telemetry_summary") or {}
    return {
        "experiment_id": record.get("experiment_id"),
        "level_id": record.get("level_id"),
        "difficulty": record.get("difficulty"),
        "title": record.get("title"),
        "backend": record.get("backend"),
        "backend_profile": record.get("backend_profile"),
        "backend_config_path": record.get("backend_config_path"),
        "model": record.get("model"),
        "thinking_enabled": record.get("thinking_enabled"),
        "repeat_index": record.get("repeat_index"),
        "status": record.get("status"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "dry_run": record.get("dry_run"),
        "success_reached_exit": record.get("success_reached_exit"),
        "termination_reason": record.get("termination_reason"),
        "call_count": record.get("call_count"),
        "valid_action_count": record.get("valid_action_count"),
        "state_change_count": record.get("state_change_count"),
        "optimal_action_count": record.get("optimal_action_count"),
        "closest_remaining_action_count": record.get("closest_remaining_action_count"),
        "progress_ratio": record.get("progress_ratio"),
        "call_count_gap_vs_optimal": record.get("call_count_gap_vs_optimal"),
        "state_change_gap_vs_optimal": record.get("state_change_gap_vs_optimal"),
        "call_efficiency_ratio": record.get("call_efficiency_ratio"),
        "state_change_efficiency_ratio": record.get("state_change_efficiency_ratio"),
        "invalid_format_count": record.get("invalid_format_count"),
        "invalid_transition_count": record.get("invalid_transition_count"),
        "hover_count": record.get("hover_count"),
        "stop_count": record.get("stop_count"),
        "stop_before_goal_count": record.get("stop_before_goal_count"),
        "unexpected_tool_call_count": record.get("unexpected_tool_call_count"),
        "parser_failure_count": record.get("parser_failure_count"),
        "truncated_step_count": record.get("truncated_step_count"),
        "nonoptimal_action_count": record.get("nonoptimal_action_count"),
        "optimal_action_match_count": record.get("optimal_action_match_count"),
        "revisited_state_count": record.get("revisited_state_count"),
        "unique_state_count": record.get("unique_state_count"),
        "max_calls": record.get("max_calls"),
        "goal_reached_call_index": record.get("goal_reached_call_index"),
        "total_latency_ms": record.get("total_latency_ms"),
        "avg_call_latency_ms": record.get("avg_call_latency_ms"),
        "avg_ttft_ms": record.get("avg_ttft_ms"),
        "total_prompt_tokens": record.get("total_prompt_tokens"),
        "total_completion_tokens": record.get("total_completion_tokens"),
        "telemetry_sample_count": telemetry.get("sample_count"),
        "telemetry_gpu_util_percent_avg": telemetry.get("gpu_util_percent_avg"),
        "telemetry_gpu_util_percent_peak": telemetry.get("gpu_util_percent_peak"),
        "telemetry_max_temp_c_peak": telemetry.get("max_temp_c_peak"),
        "telemetry_power_current_mw_avg": telemetry.get("power_current_mw_avg"),
        "telemetry_path": record.get("telemetry_path"),
        "telemetry_timeseries_path": record.get("telemetry_timeseries_path"),
        "reference_path": record.get("reference_path"),
        "step_records_path": record.get("step_records_path"),
    }


def _row_from_step(record: dict[str, Any]) -> dict[str, Any]:
    server_stats = record.get("server_stats") or {}
    return {
        "experiment_id": record.get("experiment_id"),
        "level_id": record.get("level_id"),
        "difficulty": record.get("difficulty"),
        "title": record.get("title"),
        "backend": record.get("backend"),
        "backend_profile": record.get("backend_profile"),
        "model": record.get("model"),
        "thinking_enabled": record.get("thinking_enabled"),
        "repeat_index": record.get("repeat_index"),
        "step_index": record.get("step_index"),
        "status": record.get("status"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "dry_run": record.get("dry_run"),
        "parsed_action": record.get("parsed_action"),
        "response_raw_normalized": record.get("response_raw_normalized"),
        "parsed_json": json.dumps(record.get("parsed_json"), sort_keys=True)
        if record.get("parsed_json") is not None
        else None,
        "format_valid": record.get("format_valid"),
        "action_valid": record.get("action_valid"),
        "transition_valid": record.get("transition_valid"),
        "state_changed": record.get("state_changed"),
        "goal_reached_after_step": record.get("goal_reached_after_step"),
        "terminal": record.get("terminal"),
        "termination_reason_if_terminal": record.get("termination_reason_if_terminal"),
        "outcome_reason": record.get("outcome_reason"),
        "feedback": record.get("feedback"),
        "chosen_action_is_optimal": record.get("chosen_action_is_optimal"),
        "optimal_action_count_from_state": record.get("optimal_action_count_from_state"),
        "optimal_actions_from_state": ",".join(record.get("optimal_actions_from_state") or []),
        "optimal_action_count_after_step": record.get("optimal_action_count_after_step"),
        "latency_ms": record.get("latency_ms"),
        "ttft_ms": record.get("ttft_ms"),
        "ttft_ms_effective": record.get("ttft_ms_effective"),
        "finish_reason": record.get("finish_reason"),
        "prompt_tokens": (record.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (record.get("usage") or {}).get("completion_tokens"),
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
        "server_ttft_ms": server_stats.get("server_ttft_ms"),
        "server_e2e_latency_ms": server_stats.get("server_e2e_latency_ms"),
        "server_prefill_ms": server_stats.get("server_prefill_ms"),
        "server_decode_ms": server_stats.get("server_decode_ms"),
        "server_queue_ms": server_stats.get("server_queue_ms"),
        "server_decode_tokens_per_second": server_stats.get("server_decode_tokens_per_second"),
        "artifact_prompt_path": ((record.get("artifact_paths") or {}).get("prompt_path")),
        "artifact_response_path": ((record.get("artifact_paths") or {}).get("response_path")),
        "artifact_reasoning_path": ((record.get("artifact_paths") or {}).get("reasoning_path")),
        "artifact_messages_path": ((record.get("artifact_paths") or {}).get("messages_path")),
        "artifact_prompt_token_debug_path": (
            (record.get("artifact_paths") or {}).get("prompt_token_debug_path")
        ),
        "artifact_request_payload_path": (
            (record.get("artifact_paths") or {}).get("request_payload_path")
        ),
        "artifact_raw_events_path": ((record.get("artifact_paths") or {}).get("raw_events_path")),
        "artifact_raw_event_timeline_path": (
            (record.get("artifact_paths") or {}).get("raw_event_timeline_path")
        ),
        "artifact_metrics_delta_path": (
            (record.get("artifact_paths") or {}).get("metrics_delta_path")
        ),
    }


def _aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["level_id"],
            row["difficulty"],
            row["backend"],
            row["backend_profile"],
            row["model"],
            row["thinking_enabled"],
        )
        groups[key].append(row)

    numeric_fields = [
        "call_count",
        "valid_action_count",
        "state_change_count",
        "optimal_action_count",
        "closest_remaining_action_count",
        "progress_ratio",
        "call_count_gap_vs_optimal",
        "state_change_gap_vs_optimal",
        "call_efficiency_ratio",
        "state_change_efficiency_ratio",
        "invalid_format_count",
        "invalid_transition_count",
        "hover_count",
        "stop_before_goal_count",
        "unexpected_tool_call_count",
        "parser_failure_count",
        "truncated_step_count",
        "nonoptimal_action_count",
        "optimal_action_match_count",
        "revisited_state_count",
        "unique_state_count",
        "goal_reached_call_index",
        "total_latency_ms",
        "avg_call_latency_ms",
        "avg_ttft_ms",
        "total_prompt_tokens",
        "total_completion_tokens",
        "telemetry_gpu_util_percent_avg",
        "telemetry_max_temp_c_peak",
        "telemetry_power_current_mw_avg",
    ]

    aggregates = []
    for key, bucket in sorted(groups.items()):
        level_id, difficulty, backend, backend_profile, model, thinking_enabled = key
        row = {
            "level_id": level_id,
            "difficulty": difficulty,
            "backend": backend,
            "backend_profile": backend_profile,
            "model": model,
            "thinking_enabled": thinking_enabled,
            "sample_count": len(bucket),
            "success_count": sum(1 for item in bucket if item.get("success_reached_exit")),
            "termination_reasons": ",".join(
                sorted({str(item.get("termination_reason")) for item in bucket})
            ),
            "status_set": ",".join(sorted({str(item.get("status")) for item in bucket})),
        }
        for field in numeric_fields:
            values = [item[field] for item in bucket if isinstance(item.get(field), (int, float))]
            if not values:
                continue
            row[f"{field}_mean"] = round(statistics.mean(values), 3)
            row[f"{field}_median"] = round(statistics.median(values), 3)
            row[f"{field}_min"] = round(min(values), 3)
            row[f"{field}_max"] = round(max(values), 3)
        aggregates.append(row)
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--include-dry-run", action="store_true")
    args = parser.parse_args()

    run_records_path = args.run_dir / "records.jsonl"
    step_records_path = args.run_dir / "step_records.jsonl"
    if not run_records_path.exists():
        raise SystemExit(f"Missing run records file: {run_records_path}")
    if not step_records_path.exists():
        raise SystemExit(f"Missing step records file: {step_records_path}")

    run_rows = []
    with run_records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("dry_run") and not args.include_dry_run:
                continue
            run_rows.append(_row_from_run(record))
    if not run_rows:
        raise SystemExit("No maze run records found.")

    step_rows = []
    with step_records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("dry_run") and not args.include_dry_run:
                continue
            step_rows.append(_row_from_step(record))

    summary_path = args.run_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    condition_rows = _aggregate_runs(run_rows)
    condition_path = args.run_dir / "condition_summary.csv"
    with condition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(condition_rows)

    steps_path = args.run_dir / "steps.csv"
    with steps_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)

    print(summary_path)
    print(condition_path)
    print(steps_path)


if __name__ == "__main__":
    main()
