#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.maze_navigation import run_maze_navigation_suite


def _thinking_modes(value: str) -> list[bool]:
    normalized = value.strip().lower()
    if normalized == "false":
        return [False]
    if normalized == "true":
        return [True]
    if normalized == "both":
        return [False, True]
    raise argparse.ArgumentTypeError("thinking must be one of: false, true, both")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "maze_navigation_runs",
    )
    parser.add_argument(
        "--levels-path",
        type=Path,
        default=PROJECT_ROOT / "maze_navigation" / "levels.yaml",
    )
    parser.add_argument("--difficulty", action="append", choices=["easy", "medium", "hard"])
    parser.add_argument("--level-id", action="append")
    parser.add_argument("--thinking", type=_thinking_modes, default=[False, True])
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-tegrastats", action="store_true")
    args = parser.parse_args()

    run_dir = run_maze_navigation_suite(
        project_root=PROJECT_ROOT,
        backend_config_path=args.backend_config,
        output_root=args.output_root,
        levels_path=args.levels_path,
        difficulties=set(args.difficulty) if args.difficulty else None,
        level_ids=set(args.level_id) if args.level_id else None,
        thinking_modes=args.thinking,
        repeat_count=args.repeat_count,
        max_calls_override=args.max_calls,
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
    print(run_dir)


if __name__ == "__main__":
    main()
