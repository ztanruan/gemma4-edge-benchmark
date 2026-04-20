from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


RAM_RE = re.compile(r"RAM (\d+)/(\d+)([KMG]B)")
CPU_RE = re.compile(r"CPU \[([^\]]+)\]")
CPU_UTIL_RE = re.compile(r"(\d+)%@")
GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
TEMP_RE = re.compile(r"([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C")
POWER_RE = re.compile(r"(VDD_IN|POM_5V_IN) (\d+)(mW)?/(\d+)(mW)?")


def _scale_to_mb(value: int, unit: str) -> float:
    if unit == "KB":
        return value / 1024.0
    if unit == "GB":
        return value * 1024.0
    return float(value)


def parse_tegrastats_line(line: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"raw": line.strip()}

    ram_match = RAM_RE.search(line)
    if ram_match:
        used, total, unit = ram_match.groups()
        parsed["ram_used_mb"] = _scale_to_mb(int(used), unit)
        parsed["ram_total_mb"] = _scale_to_mb(int(total), unit)

    cpu_match = CPU_RE.search(line)
    if cpu_match:
        utils = [int(value) for value in CPU_UTIL_RE.findall(cpu_match.group(1))]
        if utils:
            parsed["cpu_util_avg_percent"] = sum(utils) / len(utils)
            parsed["cpu_util_max_percent"] = max(utils)

    gpu_match = GPU_RE.search(line)
    if gpu_match:
        parsed["gpu_util_percent"] = float(gpu_match.group(1))

    temps = {name: float(value) for name, value in TEMP_RE.findall(line)}
    if temps:
        parsed["temperatures_c"] = temps
        parsed["max_temp_c"] = max(temps.values())

    power_match = POWER_RE.search(line)
    if power_match:
        rail, current, _, average, _ = power_match.groups()
        parsed["power_rail"] = rail
        parsed["power_current_mw"] = float(current)
        parsed["power_average_mw"] = float(average)

    return parsed


def summarize_tegrastats_log(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(parse_tegrastats_line(line))
    if not samples:
        return None

    def _avg(key: str) -> float | None:
        values = [sample[key] for sample in samples if key in sample]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    def _max(key: str) -> float | None:
        values = [sample[key] for sample in samples if key in sample]
        if not values:
            return None
        return round(max(values), 3)

    temperature_keys = sorted(
        {
            temp_name
            for sample in samples
            for temp_name in sample.get("temperatures_c", {})
        }
    )
    temperature_summary = {
        temp_name: {
            "avg_c": round(
                sum(sample["temperatures_c"][temp_name] for sample in samples if temp_name in sample.get("temperatures_c", {}))
                / len([sample for sample in samples if temp_name in sample.get("temperatures_c", {})]),
                3,
            ),
            "max_c": round(
                max(sample["temperatures_c"][temp_name] for sample in samples if temp_name in sample.get("temperatures_c", {})),
                3,
            ),
        }
        for temp_name in temperature_keys
    }

    return {
        "sample_count": len(samples),
        "ram_used_mb_avg": _avg("ram_used_mb"),
        "ram_used_mb_peak": _max("ram_used_mb"),
        "gpu_util_percent_avg": _avg("gpu_util_percent"),
        "gpu_util_percent_peak": _max("gpu_util_percent"),
        "cpu_util_avg_percent_avg": _avg("cpu_util_avg_percent"),
        "cpu_util_max_percent_peak": _max("cpu_util_max_percent"),
        "max_temp_c_avg": _avg("max_temp_c"),
        "max_temp_c_peak": _max("max_temp_c"),
        "power_current_mw_avg": _avg("power_current_mw"),
        "power_current_mw_peak": _max("power_current_mw"),
        "power_average_mw_avg": _avg("power_average_mw"),
        "temperature_summary": temperature_summary,
    }


def parse_tegrastats_timeseries(path: Path, interval_ms: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    series: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = parse_tegrastats_line(line)
        parsed["sample_index"] = len(series)
        parsed["offset_ms"] = len(series) * interval_ms
        series.append(parsed)
    return series


def write_tegrastats_timeseries(path: Path, interval_ms: int) -> str | None:
    series = parse_tegrastats_timeseries(path, interval_ms)
    if not series:
        return None
    timeseries_path = path.with_suffix(".jsonl")
    timeseries_path.write_text(
        "\n".join(json.dumps(item) for item in series),
        encoding="utf-8",
    )
    return str(timeseries_path)


def capture_tegrastats_snapshot(interval_ms: int = 500) -> dict[str, Any] | None:
    if shutil.which("tegrastats") is None:
        return None
    try:
        process = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None

    try:
        time.sleep(max(0.3, interval_ms / 1000.0 + 0.2))
        process.terminate()
        output, _ = process.communicate(timeout=2)
    except (subprocess.TimeoutExpired, OSError):
        process.kill()
        return None

    first_line = next((line.strip() for line in output.splitlines() if line.strip()), None)
    if not first_line:
        return None
    parsed = parse_tegrastats_line(first_line)
    parsed["raw"] = first_line
    return parsed


class TegraStatsSession:
    def __init__(self, output_path: Path, interval_ms: int = 1000):
        self.output_path = output_path
        self.interval_ms = interval_ms
        self.process: subprocess.Popen[str] | None = None
        self.handle = None
        self.enabled = shutil.which("tegrastats") is not None

    def start(self) -> bool:
        if not self.enabled:
            return False
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return True

    def stop(self) -> None:
        if not self.process:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None
        if self.handle:
            self.handle.close()
            self.handle = None
        time.sleep(0.1)
