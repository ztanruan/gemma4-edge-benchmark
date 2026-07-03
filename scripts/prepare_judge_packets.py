#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--include-dry-run", action="store_true")
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"Missing records file: {records_path}")

    project_root = Path(__file__).resolve().parents[1]
    packets_dir = args.run_dir / "judge_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    grouped_packets: dict[tuple[str, str], dict[str, Any]] = {}

    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("dry_run") and not args.include_dry_run:
                continue
            scenario_doc = project_root / "docs" / "scenarios" / f"{record['scenario_id']}.md"
            scenario_text = scenario_doc.read_text(encoding="utf-8")
            packet = {
                "scenario_id": record["scenario_id"],
                "backend": record["backend"],
                "thinking_enabled": record["thinking_enabled"],
                "repeat_index": record.get("repeat_index"),
                "status": record.get("status", "completed"),
                "dry_run": record.get("dry_run", False),
                "image_files": record.get("image_files") or [],
                "scenario_doc": scenario_text,
                "turns": record.get("turns"),
                "final_answer": record.get("final_answer"),
            }
            output_name = (
                f"{record['scenario_id']}__thinking_{str(record['thinking_enabled']).lower()}"
                f"__repeat_{int(record.get('repeat_index', 1)):02d}.json"
            )
            (packets_dir / output_name).write_text(json.dumps(packet, indent=2), encoding="utf-8")

            if record.get("review_scope") == "across_repeats":
                group_key = (
                    record["scenario_id"],
                    str(record["thinking_enabled"]).lower(),
                )
                grouped = grouped_packets.setdefault(
                    group_key,
                    {
                        "scenario_id": record["scenario_id"],
                        "backend": record["backend"],
                        "thinking_enabled": record["thinking_enabled"],
                        "review_scope": record.get("review_scope"),
                        "image_files": record.get("image_files") or [],
                        "scenario_doc": scenario_text,
                        "records": [],
                    },
                )
                grouped["records"].append(
                    {
                        "repeat_index": record.get("repeat_index"),
                        "status": record.get("status", "completed"),
                        "final_answer": record.get("final_answer"),
                        "turns": record.get("turns"),
                    }
                )

    for (scenario_id, thinking_enabled), packet in grouped_packets.items():
        output_name = f"{scenario_id}__thinking_{thinking_enabled}__grouped.json"
        (packets_dir / output_name).write_text(json.dumps(packet, indent=2), encoding="utf-8")

    print(packets_dir)


if __name__ == "__main__":
    main()
