#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.runner import run_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--thinking", choices=["true", "false", "both"], default="both")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--family", action="append")
    parser.add_argument("--use-case", action="append")
    parser.add_argument("--scenario-id", action="append")
    parser.add_argument("--with-tegrastats", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ignore-repeat-overrides", action="store_true")
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--no-fail-on-warmup-error", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=20260416)
    parser.add_argument("--seed", type=int, default=20260416)
    args = parser.parse_args()

    run_dir = run_suite(
        project_root=PROJECT_ROOT,
        backend_config_path=args.backend_config,
        thinking_mode=args.thinking,
        output_root=args.output_root,
        dry_run=args.dry_run,
        limit=args.limit,
        families=set(args.family) if args.family else None,
        use_cases=set(args.use_case) if args.use_case else None,
        scenario_ids=set(args.scenario_id) if args.scenario_id else None,
        with_tegrastats=args.with_tegrastats,
        repeats=args.repeats,
        respect_repeat_count_overrides=not args.ignore_repeat_overrides,
        warmup_count=args.warmup_count,
        fail_on_warmup_error=not args.no_fail_on_warmup_error,
        shuffle_seed=args.shuffle_seed,
        seed=args.seed,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
