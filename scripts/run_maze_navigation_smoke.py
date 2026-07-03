#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.maze_navigation import (
    load_maze_levels,
    run_maze_navigation_suite,
    validate_maze_levels,
)


def _thinking_modes(value: str) -> list[bool]:
    normalized = value.strip().lower()
    if normalized == "false":
        return [False]
    if normalized == "true":
        return [True]
    if normalized == "both":
        return [False, True]
    raise argparse.ArgumentTypeError("thinking must be one of: false, true, both")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _artifact_integrity(step_records: list[dict[str, Any]]) -> dict[str, Any]:
    required_keys = [
        "prompt_path",
        "messages_path",
        "response_path",
        "reasoning_path",
        "request_payload_path",
        "raw_events_path",
        "raw_event_timeline_path",
    ]
    checks: dict[str, dict[str, int]] = {key: {"present": 0, "missing": 0} for key in required_keys}
    for record in step_records:
        artifact_paths = record.get("artifact_paths") or {}
        for key in required_keys:
            path = artifact_paths.get(key)
            if path and Path(path).exists():
                checks[key]["present"] += 1
            else:
                checks[key]["missing"] += 1
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "maze_navigation_smoke_runs",
    )
    parser.add_argument(
        "--levels-path",
        type=Path,
        default=PROJECT_ROOT / "maze_navigation" / "levels.yaml",
    )
    parser.add_argument("--thinking", type=_thinking_modes, default=[False, True])
    parser.add_argument("--with-tegrastats", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    smoke_root = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    smoke_root.mkdir(parents=True, exist_ok=True)

    levels = load_maze_levels(args.levels_path)
    validations = validate_maze_levels(levels)
    run_dir = run_maze_navigation_suite(
        project_root=PROJECT_ROOT,
        backend_config_path=args.backend_config,
        output_root=smoke_root / "maze",
        levels_path=args.levels_path,
        difficulties={"easy", "medium", "hard"},
        thinking_modes=args.thinking,
        repeat_count=1,
        generation={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        dry_run=args.dry_run,
        with_tegrastats=args.with_tegrastats,
        seed=args.seed,
    )

    run_records = _load_jsonl(run_dir / "records.jsonl")
    step_records = _load_jsonl(run_dir / "step_records.jsonl")
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend_config": str(args.backend_config),
        "levels_path": str(args.levels_path),
        "thinking_modes": ["true" if mode else "false" for mode in args.thinking],
        "with_tegrastats": args.with_tegrastats,
        "dry_run": args.dry_run,
        "run_dir": str(run_dir),
        "selected_level_ids": [level.id for level in levels],
        "validation_summary": validations,
        "run_record_count": len(run_records),
        "step_record_count": len(step_records),
        "success_count": sum(1 for record in run_records if record.get("success_reached_exit")),
        "termination_reasons": sorted({record.get("termination_reason") for record in run_records}),
        "invalid_format_count_total": sum(
            int(record.get("invalid_format_count") or 0) for record in run_records
        ),
        "invalid_transition_count_total": sum(
            int(record.get("invalid_transition_count") or 0) for record in run_records
        ),
        "parser_failure_count_total": sum(
            int(record.get("parser_failure_count") or 0) for record in run_records
        ),
        "unexpected_tool_call_count_total": sum(
            int(record.get("unexpected_tool_call_count") or 0) for record in run_records
        ),
        "artifact_integrity": _artifact_integrity(step_records),
        "run_records_path": str(run_dir / "records.jsonl"),
        "step_records_path": str(run_dir / "step_records.jsonl"),
        "reference_path": str(run_dir / "maze_reference.json"),
        "runs": [
            {
                "experiment_id": record.get("experiment_id"),
                "level_id": record.get("level_id"),
                "difficulty": record.get("difficulty"),
                "thinking_enabled": record.get("thinking_enabled"),
                "success_reached_exit": record.get("success_reached_exit"),
                "termination_reason": record.get("termination_reason"),
                "call_count": record.get("call_count"),
                "optimal_action_count": record.get("optimal_action_count"),
                "call_count_gap_vs_optimal": record.get("call_count_gap_vs_optimal"),
                "invalid_format_count": record.get("invalid_format_count"),
                "invalid_transition_count": record.get("invalid_transition_count"),
                "step_records_path": record.get("step_records_path"),
            }
            for record in run_records
        ],
    }
    report_path = smoke_root / "maze_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(smoke_root)
    print(report_path)


if __name__ == "__main__":
    main()
