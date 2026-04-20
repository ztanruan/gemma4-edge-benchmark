#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.maze_navigation import load_maze_levels, validate_maze_levels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels-path", type=Path, default=PROJECT_ROOT / "maze_navigation" / "levels.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    levels = load_maze_levels(args.levels_path)
    validations = validate_maze_levels(levels)
    if args.json:
        print(json.dumps(validations, indent=2))
        return

    for item in validations:
        print(
            f"{item['level_id']}: difficulty={item['difficulty']} "
            f"optimal_action_count={item['optimal_action_count']} "
            f"required_optimal_actions={item['required_optimal_actions']} "
            f"optimal_path={item['optimal_path']}"
        )


if __name__ == "__main__":
    main()
