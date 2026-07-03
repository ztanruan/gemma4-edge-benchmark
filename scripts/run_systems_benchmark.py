#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.systems import run_systems_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "systems_runs"
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "systems" / "manifest.yaml")
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-tegrastats", action="store_true")
    parser.add_argument("--seed", type=int, default=20260417)
    args = parser.parse_args()

    run_dir = run_systems_suite(
        project_root=PROJECT_ROOT,
        backend_config_path=args.backend_config,
        output_root=args.output_root,
        manifest_path=args.manifest,
        experiments=set(args.experiment) if args.experiment else None,
        dry_run=args.dry_run,
        with_tegrastats=args.with_tegrastats,
        seed=args.seed,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
