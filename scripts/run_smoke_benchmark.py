#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.runner import run_suite
from gemma_vllm_benchmark.systems import run_systems_suite


def _select_smoke_scenario_ids(manifest_path: Path) -> list[str]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    selected_by_family: dict[str, str] = {}
    for scenario in manifest["scenarios"]:
        selected_by_family.setdefault(scenario["family"], scenario["id"])
    return [selected_by_family[family] for family in sorted(selected_by_family)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "smoke_runs")
    parser.add_argument("--thinking", choices=["true", "false", "both"], default="both")
    parser.add_argument("--with-tegrastats", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-workload", action="store_true")
    parser.add_argument("--skip-systems", action="store_true")
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=20260417)
    parser.add_argument("--seed", type=int, default=20260417)
    args = parser.parse_args()

    smoke_root = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    smoke_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend_config": str(args.backend_config),
        "thinking": args.thinking,
        "with_tegrastats": args.with_tegrastats,
        "dry_run": args.dry_run,
        "workload_run_dir": None,
        "systems_run_dir": None,
        "selected_workload_scenario_ids": [],
        "systems_manifest": str(PROJECT_ROOT / "systems" / "smoke_manifest.yaml"),
    }

    if not args.skip_workload:
        scenario_ids = _select_smoke_scenario_ids(PROJECT_ROOT / "benchmarks" / "manifest.yaml")
        report["selected_workload_scenario_ids"] = scenario_ids
        workload_run_dir = run_suite(
            project_root=PROJECT_ROOT,
            backend_config_path=args.backend_config,
            thinking_mode=args.thinking,
            output_root=smoke_root / "workload",
            dry_run=args.dry_run,
            scenario_ids=set(scenario_ids),
            with_tegrastats=args.with_tegrastats,
            repeats=1,
            respect_repeat_count_overrides=False,
            warmup_count=args.warmup_count,
            fail_on_warmup_error=True,
            shuffle_seed=args.shuffle_seed,
            seed=args.seed,
        )
        report["workload_run_dir"] = str(workload_run_dir)

    if not args.skip_systems:
        systems_run_dir = run_systems_suite(
            project_root=PROJECT_ROOT,
            backend_config_path=args.backend_config,
            output_root=smoke_root / "systems",
            manifest_path=PROJECT_ROOT / "systems" / "smoke_manifest.yaml",
            dry_run=args.dry_run,
            with_tegrastats=args.with_tegrastats,
            seed=args.seed + 500000,
        )
        report["systems_run_dir"] = str(systems_run_dir)

    report_path = smoke_root / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(smoke_root)
    print(report_path)


if __name__ == "__main__":
    main()
