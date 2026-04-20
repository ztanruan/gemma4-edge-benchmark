from __future__ import annotations

import json
import platform
import random
import re
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

from .tegrastats import TegraStatsSession, capture_tegrastats_snapshot, summarize_tegrastats_log, write_tegrastats_timeseries


SYSTEM_PROMPT = (
    "You are Gemma 4 running through vLLM on Jetson Thor for a benchmark. "
    "Follow the task exactly. Ground the answer only in the provided context and tool results. "
    "If the information is not present, say so plainly. Keep the final answer concise but complete."
)


@dataclass
class RunPaths:
    root: Path
    records_path: Path
    prompts_dir: Path
    responses_dir: Path
    telemetry_dir: Path
    metrics_dir: Path


def _chat_template_kwargs(thinking_enabled: bool) -> dict[str, Any]:
    return {"enable_thinking": thinking_enabled}


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_contexts(project_root: Path) -> dict[str, str]:
    corpora_root = project_root / "data" / "corpora"
    context_map: dict[str, str] = {}
    for file_path in corpora_root.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(project_root).as_posix()
            context_map[rel_path] = file_path.read_text(encoding="utf-8")
    return context_map


def ensure_run_dirs(output_root: Path, dry_run: bool = False) -> RunPaths:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    base_root = output_root / "dry_runs" if dry_run else output_root
    root = base_root / run_id
    prompts_dir = root / "prompts"
    responses_dir = root / "responses"
    telemetry_dir = root / "telemetry"
    metrics_dir = root / "metrics"
    for path in (prompts_dir, responses_dir, telemetry_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        records_path=root / "records.jsonl",
        prompts_dir=prompts_dir,
        responses_dir=responses_dir,
        telemetry_dir=telemetry_dir,
        metrics_dir=metrics_dir,
    )


def write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tokenize_debug_summary(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    if not detail:
        return None
    raw = detail.get("raw") or {}
    summary = {
        "request_type": detail.get("request_type"),
        "status_code": detail.get("status_code"),
        "ok": detail.get("ok"),
        "count": detail.get("count"),
        "error_type": detail.get("error_type"),
        "error": detail.get("error"),
    }
    if isinstance(raw, dict) and raw.get("max_model_len") is not None:
        summary["max_model_len"] = raw.get("max_model_len")
    return summary


def _backend_record_fields(backend: dict[str, Any], backend_config_path: Path) -> dict[str, Any]:
    supported_modalities = backend.get("supported_modalities") or []
    return {
        "backend": backend.get("name", "vllm"),
        "backend_config_path": str(backend_config_path),
        "backend_profile": backend.get("benchmark_profile"),
        "model": backend.get("model"),
        "configured_max_context_tokens": backend.get("max_context_tokens"),
        "max_soft_tokens": backend.get("max_soft_tokens"),
        "vision_budget_label": backend.get("vision_budget_label"),
        "prefix_caching_enabled": backend.get("prefix_caching_enabled"),
        "supported_modalities": supported_modalities,
        "audio_supported": backend.get("audio_supported"),
        "container_image": backend.get("container_image"),
        "container_image_tag": backend.get("container_image_tag"),
        "container_image_digest": backend.get("container_image_digest"),
    }


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_json(url: str, headers: dict[str, str]) -> Any | None:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def discover_models(base_url: str, api_key: str | None) -> list[dict[str, Any]]:
    payload = _get_json(f"{base_url.rstrip('/')}/v1/models", _headers(api_key))
    if not payload:
        return []
    return payload.get("data", [])


def verify_model(base_url: str, api_key: str | None, model: str) -> dict[str, Any]:
    models = discover_models(base_url, api_key)
    model_ids = [item.get("id") for item in models]
    return {
        "requested_model": model,
        "advertised_models": model_ids,
        "requested_model_found": model in model_ids if model_ids else None,
    }


def _run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def _read_text(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def scrape_metrics(base_url: str, api_key: str | None) -> str | None:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.get(f"{base_url.rstrip('/')}/metrics", headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        return response.text
    except requests.RequestException:
        return None


PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+eE0-9.]+)$")


def parse_prometheus_metrics(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_LINE.match(line.strip())
        if not match:
            continue
        metric_name, labels, value = match.groups()
        key = metric_name + (labels or "")
        try:
            parsed[key] = float(value)
        except ValueError:
            continue
    return parsed


def _sum_metric(metric_map: dict[str, float], metric_name: str) -> float:
    total = 0.0
    for key, value in metric_map.items():
        if key == metric_name or key.startswith(metric_name + "{"):
            total += value
    return total


def metrics_delta(before_text: str | None, after_text: str | None) -> dict[str, Any] | None:
    before = parse_prometheus_metrics(before_text)
    after = parse_prometheus_metrics(after_text)
    if not after:
        return None
    counter_metric_names = [
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
    ]
    gauge_metric_names = ["vllm:kv_cache_usage_perc"]
    return {
        "counter_deltas": {
            metric_name: _sum_metric(after, metric_name) - _sum_metric(before, metric_name)
            for metric_name in counter_metric_names
        },
        "gauges_before": {
            metric_name: _sum_metric(before, metric_name)
            for metric_name in gauge_metric_names
            if metric_name in before or any(key.startswith(metric_name + "{") for key in before)
        },
        "gauges_after": {
            metric_name: _sum_metric(after, metric_name)
            for metric_name in gauge_metric_names
            if metric_name in after or any(key.startswith(metric_name + "{") for key in after)
        },
    }


def _build_text_tokenize_request(
    base_url: str,
    model: str,
    text: str,
    add_special_tokens: bool,
) -> tuple[str, dict[str, Any]]:
    return (
        f"{base_url.rstrip('/')}/tokenize",
        {
            "model": model,
            "prompt": text,
            "add_special_tokens": add_special_tokens,
        },
    )


def tokenize_text_detailed(
    base_url: str,
    api_key: str | None,
    model: str,
    text: str,
    add_special_tokens: bool = True,
) -> dict[str, Any]:
    headers = _headers(api_key)
    url, payload = _build_text_tokenize_request(base_url, model, text, add_special_tokens)
    result: dict[str, Any] = {
        "backend": "vllm",
        "request_type": "text",
        "url": url,
        "model": model,
        "add_special_tokens": add_special_tokens,
        "text_chars": len(text),
        "text_preview": text[:160],
        "ok": False,
        "count": None,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result["status_code"] = response.status_code
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result
    except ValueError as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result

    tokens = data.get("tokens")
    if isinstance(tokens, list):
        result["ok"] = True
        result["count"] = len(tokens)
        result["raw"] = data
        return result
    if isinstance(data.get("count"), int):
        result["ok"] = True
        result["count"] = data["count"]
        result["raw"] = data
        return result
    result["raw"] = data
    result["error_type"] = "UnexpectedTokenizeResponse"
    result["error"] = "Tokenize response did not include tokens or count."
    return result


def _build_chat_tokenize_request(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    add_special_tokens: bool,
    add_generation_prompt: bool,
    continue_final_message: bool,
    chat_template_kwargs: dict[str, Any] | None,
    tools: list[dict[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_special_tokens": add_special_tokens,
        "add_generation_prompt": add_generation_prompt,
        "continue_final_message": continue_final_message,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if tools:
        payload["tools"] = tools
    return (f"{base_url.rstrip('/')}/tokenize", payload)


def tokenize_chat_messages_detailed(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    *,
    add_special_tokens: bool = True,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    headers = _headers(api_key)
    url, payload = _build_chat_tokenize_request(
        base_url=base_url,
        model=model,
        messages=messages,
        add_special_tokens=add_special_tokens,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        chat_template_kwargs=chat_template_kwargs,
        tools=tools,
    )
    result: dict[str, Any] = {
        "backend": "vllm",
        "request_type": "chat_messages",
        "url": url,
        "model": model,
        "message_count": len(messages),
        "add_special_tokens": add_special_tokens,
        "add_generation_prompt": add_generation_prompt,
        "continue_final_message": continue_final_message,
        "chat_template_kwargs": chat_template_kwargs,
        "tool_count": len(tools or []),
        "ok": False,
        "count": None,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result["status_code"] = response.status_code
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result
    except ValueError as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result

    tokens = data.get("tokens")
    if isinstance(tokens, list):
        result["ok"] = True
        result["count"] = len(tokens)
        result["raw"] = data
        return result
    if isinstance(data.get("count"), int):
        result["ok"] = True
        result["count"] = data["count"]
        result["raw"] = data
        return result
    result["raw"] = data
    result["error_type"] = "UnexpectedTokenizeResponse"
    result["error"] = "Tokenize response did not include tokens or count."
    return result


def tokenize_chat_messages(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    *,
    add_special_tokens: bool = True,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    result = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        add_special_tokens=add_special_tokens,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        chat_template_kwargs=chat_template_kwargs,
        tools=tools,
    )
    if result.get("ok"):
        return {"count": result["count"], "raw": result.get("raw")}
    return None


def tokenize_text(
    base_url: str,
    api_key: str | None,
    model: str,
    text: str,
    add_special_tokens: bool = True,
) -> dict[str, Any] | None:
    result = tokenize_text_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        text=text,
        add_special_tokens=add_special_tokens,
    )
    if result.get("ok"):
        return {"count": result["count"], "raw": result.get("raw")}
    return None


def _scenario_prompt_token_detail(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    *,
    thinking_enabled: bool,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    result = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs=_chat_template_kwargs(thinking_enabled),
        tools=tools,
    )
    if result.get("ok") and result.get("count") is not None:
        return {
            "count": result["count"],
            "summary": _tokenize_debug_summary(result),
        }
    return None


def _compute_end_to_end_tokens_per_second(latency_ms: float | None, completion_tokens: int | None) -> float | None:
    if latency_ms is None or latency_ms <= 0 or not completion_tokens:
        return None
    return completion_tokens / (latency_ms / 1000.0)


def _metric_average(metric_values: dict[str, Any] | None, sum_name: str, count_name: str) -> float | None:
    if not metric_values:
        return None
    counters = metric_values.get("counter_deltas") or {}
    total = counters.get(sum_name)
    count = counters.get(count_name)
    if total is None or count in (None, 0):
        return None
    return total / count


def _derive_server_stats(completion: dict[str, Any], metric_values: dict[str, Any] | None) -> dict[str, Any]:
    usage = completion.get("usage") or {}
    gauges_after = (metric_values or {}).get("gauges_after") or {}
    counters = (metric_values or {}).get("counter_deltas") or {}
    derived: dict[str, Any] = {
        "server_prompt_tokens": usage.get("prompt_tokens"),
        "server_completion_tokens": usage.get("completion_tokens"),
    }
    ttft_seconds = _metric_average(metric_values, "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count")
    e2e_seconds = _metric_average(metric_values, "vllm:e2e_request_latency_seconds_sum", "vllm:e2e_request_latency_seconds_count")
    prefill_seconds = _metric_average(metric_values, "vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count")
    decode_seconds = _metric_average(metric_values, "vllm:request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_count")
    queue_seconds = _metric_average(metric_values, "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count")
    generation_tokens = counters.get("vllm:generation_tokens_total")
    if ttft_seconds is not None:
        derived["server_ttft_ms"] = ttft_seconds * 1000.0
    if e2e_seconds is not None:
        derived["server_e2e_latency_ms"] = e2e_seconds * 1000.0
    if prefill_seconds is not None:
        derived["server_prefill_ms"] = prefill_seconds * 1000.0
    if decode_seconds is not None:
        derived["server_decode_ms"] = decode_seconds * 1000.0
    if queue_seconds is not None:
        derived["server_queue_ms"] = queue_seconds * 1000.0
    if generation_tokens and decode_seconds and decode_seconds > 0:
        derived["server_decode_tokens_per_second"] = generation_tokens / decode_seconds
    if gauges_after.get("vllm:kv_cache_usage_perc") is not None:
        derived["server_kv_cache_usage_perc"] = gauges_after["vllm:kv_cache_usage_perc"]
    return derived


def collect_run_metadata(
    run_paths: RunPaths,
    backend: dict[str, Any],
    model_verification: dict[str, Any],
    run_options: dict[str, Any],
) -> dict[str, Any]:
    tegra_snapshot = capture_tegrastats_snapshot()
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "backend": backend,
        "model_verification": model_verification,
        "run_options": run_options,
        "system": {
            "uname": _run_command(["uname", "-a"]),
            "nv_tegra_release": _read_text(Path("/etc/nv_tegra_release")),
            "nvidia_l4t_core": _run_command(["dpkg-query", "-W", "-f=${Version}", "nvidia-l4t-core"]),
            "nvidia_jetpack": _run_command(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]),
            "nvpmodel_q": _run_command(["nvpmodel", "-q"]),
            "jetson_clocks_show": _run_command(["jetson_clocks", "--show"]),
            "tegrastats_preflight": tegra_snapshot,
        },
    }
    metadata_path = run_paths.root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def update_run_metadata(run_paths: RunPaths, updater: Any) -> dict[str, Any]:
    metadata_path = run_paths.root / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    updater(metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def scenario_matches(
    scenario: dict[str, Any],
    families: set[str] | None,
    use_cases: set[str] | None,
) -> bool:
    if families and scenario["family"] not in families:
        return False
    if use_cases and scenario["use_case_id"] not in use_cases:
        return False
    return True


def _build_system_message(scenario: dict[str, Any]) -> str:
    lines = [SYSTEM_PROMPT]
    if scenario["mode"] == "agent":
        lines.append("You may call tools when current system state is needed to complete the task.")
    return " ".join(lines)


def _build_user_turn_message(
    scenario: dict[str, Any],
    context_map: dict[str, str],
    *,
    context_files: list[str],
    task: str,
    response_requirements: list[str],
    turn_index: int | None = None,
) -> str:
    if scenario.get("image_files") and not context_files:
        lines = [
            "Use the attached image as the only evidence for this task.",
            "No supporting text documents are provided.",
            "",
            "Task:",
            task,
            "",
            "Response requirements:",
        ]
        lines.extend(f"- {item}" for item in response_requirements)
        return "\n".join(lines).strip()

    lines = [
        f"Scenario ID: {scenario['id']}",
        f"Use Case: {scenario['use_case_title']}",
        f"Scenario Family: {scenario['family']}",
        f"Scenario Connectivity: {scenario['scenario_connectivity']}",
        f"Execution Mode: {scenario['execution_mode']}",
        f"Context Source: {scenario['context_source']}",
    ]
    if scenario.get("input_language"):
        lines.append(f"Input Language: {scenario['input_language']}")
    if scenario.get("expected_output_language"):
        lines.append(f"Expected Output Language: {scenario['expected_output_language']}")
    if scenario.get("language_variant"):
        lines.append(f"Language Variant: {scenario['language_variant']}")
    if turn_index is not None:
        lines.append(f"Conversation Turn: {turn_index}")
    if context_files:
        lines.extend([
            "",
            "Provided context files:",
        ])
        for rel_path in context_files:
            lines.append(f"[BEGIN FILE: {rel_path}]")
            lines.append(context_map[rel_path].strip())
            lines.append(f"[END FILE: {rel_path}]")
            lines.append("")
    elif scenario.get("image_files"):
        lines.extend([
            "",
            "Provided inputs:",
            "- One or more images are attached in this request.",
            "- No text context files are provided.",
            "- Use only visible image evidence plus the task instructions below.",
            "",
        ])
    lines.extend(["Task:", task, "", "Response requirements:"])
    lines.extend(f"- {item}" for item in response_requirements)
    return "\n".join(lines).strip()


def _build_user_message(scenario: dict[str, Any], context_map: dict[str, str]) -> str:
    return _build_user_turn_message(
        scenario,
        context_map,
        context_files=scenario["context_files"],
        task=scenario["task"],
        response_requirements=scenario["response_requirements"],
    )


def _build_user_content(project_root: Path, scenario: dict[str, Any], context_map: dict[str, str]) -> Any:
    text = _build_user_message(scenario, context_map)
    image_files = scenario.get("image_files") or []
    if not image_files:
        return text
    content: list[dict[str, Any]] = []
    for rel_path in image_files:
        abs_path = (project_root / rel_path).resolve()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"file://{abs_path}"},
                "uuid": rel_path,
            }
        )
    content.append({"type": "text", "text": text})
    return content


def _truncate_text_for_budget(text: str, target_chars: int) -> str:
    marker = "\n\n[TRUNCATED FOR CONTEXT BUDGET]"
    if len(text) <= target_chars:
        return text
    keep_chars = max(0, target_chars - len(marker))
    return text[:keep_chars].rstrip() + marker


def _estimate_prompt_budget_detail(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    *,
    thinking_enabled: bool,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int | None, dict[str, Any] | None]:
    result = _scenario_prompt_token_detail(
        base_url,
        api_key,
        model,
        messages,
        thinking_enabled=thinking_enabled,
        tools=tools,
    )
    if result:
        return int(result["count"]), result["summary"]
    return None, None


def _prepare_messages_for_request(
    project_root: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    backend_max_context_tokens: int | None,
    scenario: dict[str, Any],
    context_map: dict[str, str],
    thinking_enabled: bool,
    generation_profile: dict[str, Any],
    estimate_tokens: bool = True,
) -> dict[str, Any]:
    scenario_budget = scenario.get("max_context_tokens")
    enforce_prompt_budget = not scenario.get("skip_prompt_budget_enforcement", False)
    effective_max_context_tokens = (
        backend_max_context_tokens if backend_max_context_tokens is not None else scenario_budget
    ) if enforce_prompt_budget else None
    system_message = _build_system_message(scenario)
    working_context = {rel_path: context_map[rel_path] for rel_path in scenario["context_files"]}
    truncated_context_files: list[str] = []
    original_prompt_tokens = None
    prompt_token_debug = None

    def _messages_from_contexts(local_context: dict[str, str]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": _build_user_content(project_root, scenario, local_context)},
        ]

    messages = _messages_from_contexts(working_context)
    prompt_tokens = None
    if estimate_tokens and enforce_prompt_budget:
        prompt_tokens, prompt_token_debug = _estimate_prompt_budget_detail(
            base_url,
            api_key,
            model,
            messages,
            thinking_enabled=thinking_enabled,
            tools=scenario["tools"] if scenario["mode"] == "agent" else None,
        )
        original_prompt_tokens = prompt_tokens
    else:
        original_prompt_tokens = None

    if not estimate_tokens or not enforce_prompt_budget:
        return {
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_token_debug": prompt_token_debug,
            "prompt_truncated": False,
            "truncated_context_files": truncated_context_files,
            "effective_max_context_tokens": effective_max_context_tokens,
            "scenario_declared_max_context_tokens": scenario_budget,
            "context_budget_valid": None,
        }

    if effective_max_context_tokens is None:
        return {
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_token_debug": prompt_token_debug,
            "prompt_truncated": False,
            "truncated_context_files": truncated_context_files,
            "effective_max_context_tokens": effective_max_context_tokens,
            "scenario_declared_max_context_tokens": scenario_budget,
            "context_budget_valid": None,
        }

    allowed_prompt_tokens = effective_max_context_tokens - generation_profile["max_tokens"]
    if allowed_prompt_tokens <= 0:
        raise ValueError(
            f"Configured max_context_tokens={effective_max_context_tokens} leaves no room for max_tokens={generation_profile['max_tokens']}."
        )
    if prompt_tokens is None:
        raise RuntimeError(
            f"Unable to estimate prompt tokens for {scenario['id']}; cannot enforce max_context_tokens={effective_max_context_tokens}."
        )
    if prompt_tokens <= allowed_prompt_tokens:
        return {
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_token_debug": prompt_token_debug,
            "prompt_truncated": False,
            "truncated_context_files": truncated_context_files,
            "effective_max_context_tokens": effective_max_context_tokens,
            "scenario_declared_max_context_tokens": scenario_budget,
            "context_budget_valid": True,
        }

    context_files = list(scenario["context_files"])
    for _attempt in range(12):
        if prompt_tokens is not None and prompt_tokens <= allowed_prompt_tokens:
            break
        reduction_ratio = max(0.05, min(0.98, (allowed_prompt_tokens / max(prompt_tokens or 1, 1)) * 0.97))
        changed = False
        for rel_path in sorted(context_files, key=lambda item: len(working_context[item]), reverse=True):
            current_text = working_context[rel_path]
            min_chars = 256
            target_chars = max(min_chars, int(len(current_text) * reduction_ratio))
            if target_chars < len(current_text):
                working_context[rel_path] = _truncate_text_for_budget(current_text, target_chars)
                if rel_path not in truncated_context_files:
                    truncated_context_files.append(rel_path)
                changed = True
        if not changed:
            break
        messages = _messages_from_contexts(working_context)
        prompt_tokens, prompt_token_debug = _estimate_prompt_budget_detail(
            base_url,
            api_key,
            model,
            messages,
            thinking_enabled=thinking_enabled,
            tools=scenario["tools"] if scenario["mode"] == "agent" else None,
        )

    if prompt_tokens is None or prompt_tokens + generation_profile["max_tokens"] > effective_max_context_tokens:
        raise ValueError(
            f"Scenario {scenario['id']} exceeds the configured context budget after truncation: "
            f"prompt_tokens={prompt_tokens}, max_output_tokens={generation_profile['max_tokens']}, "
            f"max_context_tokens={effective_max_context_tokens}."
        )

    return {
        "messages": messages,
        "prompt_tokens": prompt_tokens,
        "original_prompt_tokens": original_prompt_tokens,
        "prompt_token_debug": prompt_token_debug,
        "prompt_truncated": True,
        "truncated_context_files": truncated_context_files,
        "effective_max_context_tokens": effective_max_context_tokens,
        "scenario_declared_max_context_tokens": scenario_budget,
        "context_budget_valid": True,
    }


def _prepare_messages_for_conversation_turn(
    base_url: str,
    api_key: str | None,
    model: str,
    backend_max_context_tokens: int | None,
    scenario: dict[str, Any],
    context_map: dict[str, str],
    thinking_enabled: bool,
    generation_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]],
    turn: dict[str, Any],
    turn_index: int,
    estimate_tokens: bool = True,
) -> dict[str, Any]:
    scenario_budget = scenario.get("max_context_tokens")
    effective_max_context_tokens = backend_max_context_tokens if backend_max_context_tokens is not None else scenario_budget
    context_files = turn.get("context_files") or scenario["context_files"]
    response_requirements = turn.get("response_requirements") or scenario["response_requirements"]
    working_context = {rel_path: context_map[rel_path] for rel_path in context_files}
    truncated_context_files: list[str] = []
    original_prompt_tokens = None
    prompt_token_debug = None

    def _messages_from_contexts(local_context: dict[str, str]) -> list[dict[str, Any]]:
        return conversation_history + [
            {
                "role": "user",
                "content": _build_user_turn_message(
                    scenario,
                    local_context,
                    context_files=context_files,
                    task=turn["task"],
                    response_requirements=response_requirements,
                    turn_index=turn_index,
                ),
            }
        ]

    messages = _messages_from_contexts(working_context)
    prompt_tokens = None
    if estimate_tokens:
        prompt_tokens, prompt_token_debug = _estimate_prompt_budget_detail(
            base_url,
            api_key,
            model,
            messages,
            thinking_enabled=thinking_enabled,
            tools=scenario["tools"] if scenario["mode"] == "agent" else None,
        )
        original_prompt_tokens = prompt_tokens

    if not estimate_tokens:
        return {
            "messages": messages,
            "prompt_tokens": None,
            "original_prompt_tokens": None,
            "prompt_token_debug": None,
            "prompt_truncated": False,
            "truncated_context_files": truncated_context_files,
            "effective_max_context_tokens": effective_max_context_tokens,
            "scenario_declared_max_context_tokens": scenario_budget,
        }

    if effective_max_context_tokens is None:
        return {
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_token_debug": prompt_token_debug,
            "prompt_truncated": False,
            "truncated_context_files": truncated_context_files,
            "effective_max_context_tokens": effective_max_context_tokens,
            "scenario_declared_max_context_tokens": scenario_budget,
        }

    allowed_prompt_tokens = effective_max_context_tokens - generation_profile["max_tokens"]
    if allowed_prompt_tokens <= 0:
        raise ValueError(
            f"Configured max_context_tokens={effective_max_context_tokens} leaves no room for max_tokens={generation_profile['max_tokens']}."
        )
    if prompt_tokens is None:
        raise RuntimeError(
            f"Unable to estimate prompt tokens for {scenario['id']} turn {turn_index}; cannot enforce max_context_tokens={effective_max_context_tokens}."
        )
    if prompt_tokens <= allowed_prompt_tokens:
        return {
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_token_debug": prompt_token_debug,
            "prompt_truncated": False,
            "truncated_context_files": truncated_context_files,
            "effective_max_context_tokens": effective_max_context_tokens,
            "scenario_declared_max_context_tokens": scenario_budget,
        }

    for _attempt in range(12):
        if prompt_tokens is not None and prompt_tokens <= allowed_prompt_tokens:
            break
        reduction_ratio = max(0.05, min(0.98, (allowed_prompt_tokens / max(prompt_tokens or 1, 1)) * 0.97))
        changed = False
        for rel_path in sorted(context_files, key=lambda item: len(working_context[item]), reverse=True):
            current_text = working_context[rel_path]
            min_chars = 256
            target_chars = max(min_chars, int(len(current_text) * reduction_ratio))
            if target_chars < len(current_text):
                working_context[rel_path] = _truncate_text_for_budget(current_text, target_chars)
                if rel_path not in truncated_context_files:
                    truncated_context_files.append(rel_path)
                changed = True
        if not changed:
            break
        messages = _messages_from_contexts(working_context)
        prompt_tokens, prompt_token_debug = _estimate_prompt_budget_detail(
            base_url,
            api_key,
            model,
            messages,
            thinking_enabled=thinking_enabled,
            tools=scenario["tools"] if scenario["mode"] == "agent" else None,
        )

    if prompt_tokens is None or prompt_tokens + generation_profile["max_tokens"] > effective_max_context_tokens:
        raise ValueError(
            f"Scenario {scenario['id']} turn {turn_index} exceeds the configured context budget after truncation: "
            f"prompt_tokens={prompt_tokens}, max_output_tokens={generation_profile['max_tokens']}, "
            f"max_context_tokens={effective_max_context_tokens}."
        )

    return {
        "messages": messages,
        "prompt_tokens": prompt_tokens,
        "original_prompt_tokens": original_prompt_tokens,
        "prompt_token_debug": prompt_token_debug,
        "prompt_truncated": True,
        "truncated_context_files": truncated_context_files,
        "effective_max_context_tokens": effective_max_context_tokens,
        "scenario_declared_max_context_tokens": scenario_budget,
    }


def _extract_stream_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _parse_json_maybe(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _normalize_tool_calls(tool_calls_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index in sorted(tool_calls_map):
        call = tool_calls_map[index]
        arguments = call["function"].get("arguments", "")
        normalized.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": call.get("type", "function"),
                "function": {
                    "name": call["function"].get("name", ""),
                    "arguments": arguments,
                    "parsed_arguments": _parse_json_maybe(arguments),
                },
            }
        )
    return normalized


def _tool_calls_fallback_text(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return ""
    return json.dumps(tool_calls, sort_keys=True)


def _detect_empty_stream_bug(
    content: str,
    reasoning: str,
    tool_calls: list[dict[str, Any]],
    data_event_count: int,
) -> dict[str, Any] | None:
    if content.strip() or reasoning.strip() or tool_calls:
        return None
    if data_event_count == 0:
        return None
    return {
        "detected": True,
        "reason": "vllm_chat_stream_empty_payload",
        "message": "vLLM returned streamed chat data events but no content, reasoning, or tool calls.",
        "data_event_count": data_event_count,
    }


def _detect_reasoning_only_truncated(
    content: str,
    reasoning: str,
    tool_calls: list[dict[str, Any]],
    finish_reason: str | None,
    data_event_count: int,
) -> dict[str, Any] | None:
    if content.strip() or tool_calls:
        return None
    if not reasoning.strip():
        return None
    if finish_reason != "length":
        return None
    if data_event_count == 0:
        return None
    return {
        "detected": True,
        "reason": "reasoning_only_truncated",
        "message": "The model emitted reasoning but hit the max token limit before producing visible answer content.",
        "finish_reason": finish_reason,
        "data_event_count": data_event_count,
    }


def _completion_status(completion: dict[str, Any]) -> str:
    if completion.get("reasoning_only_truncated"):
        return "reasoning_only_truncated"
    if completion.get("empty_stream_bug"):
        return "empty_stream_bug"
    return "completed"


def request_chat_completion(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    generation: dict[str, Any],
    thinking_enabled: bool,
    prompt_token_estimate: int | None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    seed: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_include_usage": True,
        "max_tokens": generation["max_tokens"],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "chat_template_kwargs": _chat_template_kwargs(thinking_enabled),
    }
    if seed is not None:
        payload["seed"] = seed
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None:
        payload["response_format"] = response_format

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    start = time.perf_counter()
    response = requests.post(url, headers=_headers(api_key), json=payload, stream=True, timeout=600)
    response.raise_for_status()

    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    raw_events: list[str] = []
    raw_event_records: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    first_output_ms: float | None = None
    last_event: dict[str, Any] | None = None
    tool_calls_map: dict[int, dict[str, Any]] = {}
    data_event_count = 0
    content_event_count = 0
    reasoning_event_count = 0
    tool_call_event_count = 0
    output_event_offsets_ms: list[float] = []
    completion_token_fallback_debug: dict[str, Any] | None = None

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        event_offset_ms = (time.perf_counter() - start) * 1000.0
        raw_events.append(line)
        raw_event_records.append({"offset_ms": round(event_offset_ms, 3), "line": line})
        if not line.startswith("data:"):
            continue
        data = line[5:].lstrip()
        if data == "[DONE]":
            break
        data_event_count += 1
        event = json.loads(data)
        last_event = event
        if "error" in event:
            raise RuntimeError(json.dumps(event["error"]))
        choices = event.get("choices", [])
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}

        content_delta = _extract_stream_text(delta.get("content"))
        reasoning_delta = delta.get("reasoning")
        if reasoning_delta is None:
            reasoning_delta = delta.get("reasoning_content")
        reasoning_delta = _extract_stream_text(reasoning_delta)
        tool_call_deltas = delta.get("tool_calls") or []

        if content_delta:
            content_chunks.append(content_delta)
            content_event_count += 1
        if reasoning_delta:
            reasoning_chunks.append(reasoning_delta)
            reasoning_event_count += 1
        if tool_call_deltas:
            tool_call_event_count += 1
            for tool_delta in tool_call_deltas:
                index = tool_delta.get("index", 0)
                current = tool_calls_map.setdefault(
                    index,
                    {"id": None, "type": tool_delta.get("type", "function"), "function": {"name": "", "arguments": ""}},
                )
                if tool_delta.get("id"):
                    current["id"] = tool_delta["id"]
                if tool_delta.get("type"):
                    current["type"] = tool_delta["type"]
                function = tool_delta.get("function") or {}
                if function.get("name"):
                    current["function"]["name"] += function["name"]
                if function.get("arguments"):
                    current["function"]["arguments"] += function["arguments"]

        if content_delta or reasoning_delta or tool_call_deltas:
            output_event_offsets_ms.append(round(event_offset_ms, 3))
        if first_output_ms is None and (content_delta or reasoning_delta or tool_call_deltas):
            first_output_ms = event_offset_ms

        finish_reason = choice.get("finish_reason") or finish_reason
        usage = event.get("usage") or usage

    latency_ms = (time.perf_counter() - start) * 1000.0
    content = "".join(content_chunks)
    reasoning = "".join(reasoning_chunks)
    tool_calls = _normalize_tool_calls(tool_calls_map)

    usage = dict(usage or {})
    if prompt_token_estimate is not None and usage.get("prompt_tokens") is None:
        usage["prompt_tokens"] = prompt_token_estimate
        usage["prompt_tokens_source"] = "tokenize_prompt_estimate"
    elif usage.get("prompt_tokens") is not None:
        usage.setdefault("prompt_tokens_source", "server_usage")

    reported_completion_tokens = usage.get("completion_tokens")
    if reported_completion_tokens is not None:
        usage.setdefault("completion_tokens_server_reported", reported_completion_tokens)
        usage.setdefault("completion_tokens_source", "server_usage")

    completion_tokens = reported_completion_tokens
    assistant_fallback_text = "\n".join(part for part in [reasoning, content, _tool_calls_fallback_text(tool_calls)] if part)
    if completion_tokens in (None, 0) and assistant_fallback_text.strip():
        completion_token_fallback_debug = tokenize_text_detailed(
            base_url=base_url,
            api_key=api_key,
            model=model,
            text=assistant_fallback_text,
            add_special_tokens=False,
        )
        if completion_token_fallback_debug.get("count") is not None:
            completion_tokens = completion_token_fallback_debug["count"]
            usage["completion_tokens"] = completion_tokens
            usage["completion_tokens_source"] = "tokenize_completion_fallback"

    end_to_end_tokens_per_second = _compute_end_to_end_tokens_per_second(latency_ms, completion_tokens)
    reasoning_only_truncated = _detect_reasoning_only_truncated(
        content,
        reasoning,
        tool_calls,
        finish_reason,
        data_event_count,
    )
    empty_stream_bug = _detect_empty_stream_bug(content, reasoning, tool_calls, data_event_count)

    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "ttft_ms": first_output_ms,
        "latency_ms": latency_ms,
        "usage": usage,
        "finish_reason": finish_reason,
        "raw_events": raw_events,
        "raw_event_records": raw_event_records,
        "end_to_end_tokens_per_second": end_to_end_tokens_per_second,
        "server_decode_tokens_per_second": None,
        "last_event": last_event,
        "completion_token_fallback_debug": completion_token_fallback_debug,
        "data_event_count": data_event_count,
        "content_event_count": content_event_count,
        "non_empty_text_event_count": content_event_count,
        "reasoning_event_count": reasoning_event_count,
        "tool_call_event_count": tool_call_event_count,
        "output_event_offsets_ms": output_event_offsets_ms,
        "reasoning_only_truncated": reasoning_only_truncated,
        "empty_stream_bug": empty_stream_bug,
        "request_payload": payload,
    }


def _assistant_message_from_completion(completion: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    if completion["content"]:
        message["content"] = completion["content"]
    else:
        message["content"] = None
    if completion["tool_calls"]:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": call["type"],
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                },
            }
            for call in completion["tool_calls"]
        ]
    return message


def _compare_expected_tool_calls(
    actual_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for index, call in enumerate(actual_calls):
        expected = expected_calls[index] if index < len(expected_calls) else None
        parsed_args = call["function"].get("parsed_arguments")
        call_name = call["function"]["name"]
        comparison = {
            "index": index,
            "actual_name": call_name,
            "actual_arguments": parsed_args,
            "expected_name": expected.get("name") if expected else None,
            "expected_arguments": expected.get("arguments") if expected else None,
        }
        comparison["name_match"] = expected is not None and call_name == expected.get("name")
        comparison["arguments_match"] = expected is not None and parsed_args == expected.get("arguments")
        comparisons.append(comparison)
    return {"expected_count": len(expected_calls), "actual_count": len(actual_calls), "comparisons": comparisons}


def _resolve_tool_results(
    actual_calls: list[dict[str, Any]],
    remaining_results: list[dict[str, Any]],
) -> dict[str, Any]:
    used_indexes: set[int] = set()
    matched: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []

    for call in actual_calls:
        name = call["function"]["name"]
        match_index = None
        for idx, result in enumerate(remaining_results):
            if idx in used_indexes:
                continue
            if result["name"] == name:
                match_index = idx
                break
        if match_index is None:
            unresolved.append(call)
            continue
        used_indexes.add(match_index)
        matched_result = remaining_results[match_index]
        matched.append(matched_result)
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(matched_result["response"], sort_keys=True),
            }
        )

    remaining = [result for idx, result in enumerate(remaining_results) if idx not in used_indexes]
    return {
        "matched_results": matched,
        "unresolved_calls": unresolved,
        "tool_messages": tool_messages,
        "remaining_results": remaining,
    }


def _write_metrics_snapshot(path: Path, contents: str | None) -> None:
    if contents is not None:
        path.write_text(contents, encoding="utf-8")


def _record_non_agent_turn(
    run_paths: RunPaths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    scenario: dict[str, Any],
    messages: list[dict[str, Any]],
    prompt_name: str,
    profile: dict[str, Any],
    prompt_tokens: int | None,
    prompt_token_debug: dict[str, Any] | None,
    prompt_truncated: bool,
    original_prompt_tokens: int | None,
    truncated_context_files: list[str],
    effective_max_context_tokens: int | None,
    scenario_declared_max_context_tokens: int | None,
    context_budget_valid: bool | None,
    thinking_enabled: bool,
    seed: int | None,
    repeat_index: int,
    with_tegrastats: bool,
) -> dict[str, Any]:
    response_base = run_paths.responses_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}"
    response_base.mkdir(parents=True, exist_ok=True)
    _write_json(response_base / "messages.json", messages)

    telemetry_path = run_paths.telemetry_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}.log"
    tegra = TegraStatsSession(telemetry_path)
    tegra_active = False
    if with_tegrastats:
        tegra_active = tegra.start()

    metrics_before = scrape_metrics(base_url, api_key)
    _write_metrics_snapshot(run_paths.metrics_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}__before.prom", metrics_before)

    try:
            completion = request_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=backend_fields["model"],
                messages=messages,
                generation=profile,
                thinking_enabled=thinking_enabled,
                prompt_token_estimate=prompt_tokens,
                response_format=scenario.get("response_format"),
                seed=seed,
            )
    finally:
        if tegra_active:
            tegra.stop()

    metrics_after = scrape_metrics(base_url, api_key)
    _write_metrics_snapshot(run_paths.metrics_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}__after.prom", metrics_after)
    metric_values = metrics_delta(metrics_before, metrics_after)

    telemetry_summary = summarize_tegrastats_log(telemetry_path) if tegra_active else None
    telemetry_timeseries_path = write_tegrastats_timeseries(telemetry_path, tegra.interval_ms) if tegra_active else None
    server_stats = _derive_server_stats(completion, metric_values)
    effective_ttft_ms = completion["ttft_ms"] if completion["ttft_ms"] is not None else server_stats.get("server_ttft_ms")
    events_path = response_base / "response_events.sse"
    event_timeline_path = response_base / "response_events_timeline.jsonl"
    completion_token_debug_path = response_base / "completion_token_fallback_debug.json"
    request_payload_path = response_base / "request_payload.json"
    prompt_token_debug_path = response_base / "prompt_token_debug.json"

    _write_text(response_base / "response.txt", completion["content"])
    _write_text(response_base / "reasoning.txt", completion["reasoning"])
    _write_json(response_base / "response_tool_calls.json", completion["tool_calls"])
    _write_text(events_path, "\n".join(completion["raw_events"]))
    _write_text(event_timeline_path, "\n".join(json.dumps(item) for item in completion["raw_event_records"]))
    _write_json(request_payload_path, completion["request_payload"])
    if prompt_token_debug is not None:
        _write_json(prompt_token_debug_path, prompt_token_debug)
    if completion["completion_token_fallback_debug"] is not None:
        _write_json(completion_token_debug_path, completion["completion_token_fallback_debug"])

    status = _completion_status(completion)

    return {
        "scenario_id": scenario["id"],
        "use_case_id": scenario["use_case_id"],
        "use_case_title": scenario["use_case_title"],
        "family": scenario["family"],
        "mode": scenario["mode"],
        "scenario_connectivity": scenario["scenario_connectivity"],
        "execution_mode": scenario["execution_mode"],
        "context_source": scenario["context_source"],
        "input_language": scenario.get("input_language"),
        "expected_output_language": scenario.get("expected_output_language"),
        "language_variant": scenario.get("language_variant"),
        "image_files": scenario.get("image_files") or [],
        "review_scope": scenario.get("review_scope", "single_response"),
        "variant_id": scenario.get("variant_id"),
        **backend_fields,
        "thinking_enabled": thinking_enabled,
        "repeat_index": repeat_index,
        "generation_profile": scenario["generation_profile"],
        "generation": profile,
        "seed": seed,
        "dry_run": False,
        "max_context_tokens": effective_max_context_tokens,
        "scenario_declared_max_context_tokens": scenario_declared_max_context_tokens,
        "context_budget_valid": context_budget_valid,
        "prompt_truncated": prompt_truncated,
        "original_prompt_token_estimate": original_prompt_tokens,
        "truncated_context_files": truncated_context_files,
        "prompt_path": str(run_paths.prompts_dir / prompt_name),
        "initial_prompt_token_estimate": prompt_tokens,
        "initial_prompt_token_debug": prompt_token_debug,
        "initial_prompt_token_debug_path": str(prompt_token_debug_path) if prompt_token_debug is not None else None,
        "status": status,
        "turns": [
            {
                "turn_index": 1,
                "prompt_token_estimate": prompt_tokens,
                "prompt_token_debug": prompt_token_debug,
                "prompt_token_debug_path": str(prompt_token_debug_path) if prompt_token_debug is not None else None,
                "latency_ms": completion["latency_ms"],
                "ttft_ms": completion["ttft_ms"],
                "ttft_ms_effective": effective_ttft_ms,
                "ttft_source": "client_stream" if completion["ttft_ms"] is not None else ("server_metrics" if effective_ttft_ms is not None else None),
                "end_to_end_tokens_per_second": completion["end_to_end_tokens_per_second"],
                "usage": completion["usage"],
                "finish_reason": completion["finish_reason"],
                "raw_text": completion["content"],
                "raw_reasoning": completion["reasoning"],
                "raw_event_count": len(completion["raw_events"]),
                "data_event_count": completion["data_event_count"],
                "content_event_count": completion["content_event_count"],
                "non_empty_text_event_count": completion["non_empty_text_event_count"],
                "reasoning_event_count": completion["reasoning_event_count"],
                "tool_call_event_count": completion["tool_call_event_count"],
                "raw_events_path": str(events_path),
                "raw_event_timeline_path": str(event_timeline_path),
                "request_payload_path": str(request_payload_path),
                "parsed_thought": completion["reasoning"],
                "parsed_answer": completion["content"],
                "parsed_tool_calls": completion["tool_calls"],
                "server_metrics_delta": metric_values,
                "server_stats": server_stats,
                "completion_token_fallback_debug": completion["completion_token_fallback_debug"],
                "completion_token_fallback_debug_path": str(completion_token_debug_path) if completion["completion_token_fallback_debug"] is not None else None,
                "reasoning_only_truncated": completion["reasoning_only_truncated"],
                "empty_stream_bug": completion["empty_stream_bug"],
            }
        ],
        "final_answer": completion["content"],
        "total_latency_ms": completion["latency_ms"],
        "telemetry_path": str(telemetry_path) if tegra_active else None,
        "telemetry_timeseries_path": telemetry_timeseries_path,
        "telemetry_summary": telemetry_summary,
    }


def _record_agent_turns(
    run_paths: RunPaths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    scenario: dict[str, Any],
    initial_messages: list[dict[str, Any]],
    prompt_name: str,
    profile: dict[str, Any],
    prompt_tokens: int | None,
    prompt_token_debug: dict[str, Any] | None,
    prompt_truncated: bool,
    original_prompt_tokens: int | None,
    truncated_context_files: list[str],
    effective_max_context_tokens: int | None,
    scenario_declared_max_context_tokens: int | None,
    context_budget_valid: bool | None,
    thinking_enabled: bool,
    seed: int | None,
    repeat_index: int,
    with_tegrastats: bool,
) -> dict[str, Any]:
    response_base = run_paths.responses_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}"
    response_base.mkdir(parents=True, exist_ok=True)
    _write_json(response_base / "turn_01_messages.json", initial_messages)

    telemetry_path = run_paths.telemetry_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}.log"
    tegra = TegraStatsSession(telemetry_path)
    tegra_active = False
    if with_tegrastats:
        tegra_active = tegra.start()

    turns: list[dict[str, Any]] = []
    messages = [json.loads(json.dumps(message)) for message in initial_messages]
    remaining_results = list(scenario["tool_results"])
    expected_tool_calls = list(scenario.get("expected_tool_calls", []))
    total_latency_ms = 0.0
    final_answer = ""
    status = "completed"
    initial_prompt_token_debug_path = response_base / "turn_01_prompt_token_debug.json"
    if prompt_token_debug is not None:
        _write_json(initial_prompt_token_debug_path, prompt_token_debug)

    try:
        for turn_index in range(1, scenario.get("max_agent_turns", 4) + 1):
            current_prompt_tokens, current_prompt_token_debug = _estimate_prompt_budget_detail(
                base_url,
                api_key,
                backend_fields["model"],
                messages,
                thinking_enabled=thinking_enabled,
                tools=scenario["tools"],
            )
            metrics_before = scrape_metrics(base_url, api_key)
            _write_metrics_snapshot(
                run_paths.metrics_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}__turn_{turn_index:02d}__before.prom",
                metrics_before,
            )

            completion = request_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=backend_fields["model"],
                messages=messages,
                generation=profile,
                thinking_enabled=thinking_enabled,
                prompt_token_estimate=current_prompt_tokens,
                tools=scenario["tools"],
                tool_choice="auto",
                seed=None if seed is None else seed + turn_index - 1,
            )

            metrics_after = scrape_metrics(base_url, api_key)
            _write_metrics_snapshot(
                run_paths.metrics_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}__turn_{turn_index:02d}__after.prom",
                metrics_after,
            )
            metric_values = metrics_delta(metrics_before, metrics_after)

            server_stats = _derive_server_stats(completion, metric_values)
            effective_ttft_ms = completion["ttft_ms"] if completion["ttft_ms"] is not None else server_stats.get("server_ttft_ms")
            events_path = response_base / f"turn_{turn_index:02d}_events.sse"
            event_timeline_path = response_base / f"turn_{turn_index:02d}_events_timeline.jsonl"
            completion_token_debug_path = response_base / f"turn_{turn_index:02d}_completion_token_fallback_debug.json"
            request_payload_path = response_base / f"turn_{turn_index:02d}_request_payload.json"
            prompt_token_debug_path = response_base / f"turn_{turn_index:02d}_prompt_token_debug.json"
            _write_text(response_base / f"turn_{turn_index:02d}_response.txt", completion["content"])
            _write_text(response_base / f"turn_{turn_index:02d}_reasoning.txt", completion["reasoning"])
            _write_json(response_base / f"turn_{turn_index:02d}_tool_calls.json", completion["tool_calls"])
            _write_text(events_path, "\n".join(completion["raw_events"]))
            _write_text(event_timeline_path, "\n".join(json.dumps(item) for item in completion["raw_event_records"]))
            _write_json(request_payload_path, completion["request_payload"])
            if current_prompt_token_debug is not None:
                _write_json(prompt_token_debug_path, current_prompt_token_debug)
            if completion["completion_token_fallback_debug"] is not None:
                _write_json(completion_token_debug_path, completion["completion_token_fallback_debug"])

            expected_comparison = _compare_expected_tool_calls(completion["tool_calls"], expected_tool_calls)
            resolution = _resolve_tool_results(completion["tool_calls"], remaining_results)
            turn_record = {
                "turn_index": turn_index,
                "prompt_token_estimate": current_prompt_tokens,
                "prompt_token_debug": current_prompt_token_debug,
                "prompt_token_debug_path": str(prompt_token_debug_path) if current_prompt_token_debug is not None else None,
                "latency_ms": completion["latency_ms"],
                "ttft_ms": completion["ttft_ms"],
                "ttft_ms_effective": effective_ttft_ms,
                "ttft_source": "client_stream" if completion["ttft_ms"] is not None else ("server_metrics" if effective_ttft_ms is not None else None),
                "end_to_end_tokens_per_second": completion["end_to_end_tokens_per_second"],
                "usage": completion["usage"],
                "finish_reason": completion["finish_reason"],
                "raw_text": completion["content"],
                "raw_reasoning": completion["reasoning"],
                "raw_event_count": len(completion["raw_events"]),
                "data_event_count": completion["data_event_count"],
                "content_event_count": completion["content_event_count"],
                "non_empty_text_event_count": completion["non_empty_text_event_count"],
                "reasoning_event_count": completion["reasoning_event_count"],
                "tool_call_event_count": completion["tool_call_event_count"],
                "raw_events_path": str(events_path),
                "raw_event_timeline_path": str(event_timeline_path),
                "request_payload_path": str(request_payload_path),
                "parsed_thought": completion["reasoning"],
                "parsed_answer": completion["content"],
                "parsed_tool_calls": completion["tool_calls"],
                "expected_tool_call_comparison": expected_comparison,
                "tool_resolution": resolution,
                "server_metrics_delta": metric_values,
                "server_stats": server_stats,
                "completion_token_fallback_debug": completion["completion_token_fallback_debug"],
                "completion_token_fallback_debug_path": str(completion_token_debug_path) if completion["completion_token_fallback_debug"] is not None else None,
                "reasoning_only_truncated": completion["reasoning_only_truncated"],
                "empty_stream_bug": completion["empty_stream_bug"],
            }
            turns.append(turn_record)
            total_latency_ms += completion["latency_ms"]

            if completion["reasoning_only_truncated"]:
                status = "reasoning_only_truncated"
                final_answer = completion["content"]
                break

            if completion["empty_stream_bug"]:
                status = "empty_stream_bug"
                final_answer = completion["content"]
                break

            if completion["tool_calls"]:
                if resolution["unresolved_calls"]:
                    status = "unresolved_tool_call"
                    final_answer = completion["content"]
                    break
                messages.append(_assistant_message_from_completion(completion))
                messages.extend(resolution["tool_messages"])
                remaining_results = resolution["remaining_results"]
                _write_json(response_base / f"turn_{turn_index:02d}_continued_messages.json", messages)
                continue

            final_answer = completion["content"]
            break
        else:
            status = "max_agent_turns_exhausted"
    finally:
        if tegra_active:
            tegra.stop()

    telemetry_summary = summarize_tegrastats_log(telemetry_path) if tegra_active else None
    telemetry_timeseries_path = write_tegrastats_timeseries(telemetry_path, tegra.interval_ms) if tegra_active else None
    return {
        "scenario_id": scenario["id"],
        "use_case_id": scenario["use_case_id"],
        "use_case_title": scenario["use_case_title"],
        "family": scenario["family"],
        "mode": scenario["mode"],
        "scenario_connectivity": scenario["scenario_connectivity"],
        "execution_mode": scenario["execution_mode"],
        "context_source": scenario["context_source"],
        "input_language": scenario.get("input_language"),
        "expected_output_language": scenario.get("expected_output_language"),
        "language_variant": scenario.get("language_variant"),
        "image_files": scenario.get("image_files") or [],
        "review_scope": scenario.get("review_scope", "single_response"),
        "variant_id": scenario.get("variant_id"),
        **backend_fields,
        "thinking_enabled": thinking_enabled,
        "repeat_index": repeat_index,
        "generation_profile": scenario["generation_profile"],
        "generation": profile,
        "seed": seed,
        "dry_run": False,
        "max_context_tokens": effective_max_context_tokens,
        "scenario_declared_max_context_tokens": scenario_declared_max_context_tokens,
        "context_budget_valid": context_budget_valid,
        "prompt_truncated": prompt_truncated,
        "original_prompt_token_estimate": original_prompt_tokens,
        "truncated_context_files": truncated_context_files,
        "prompt_path": str(run_paths.prompts_dir / prompt_name),
        "initial_prompt_token_estimate": prompt_tokens,
        "initial_prompt_token_debug": prompt_token_debug,
        "initial_prompt_token_debug_path": str(initial_prompt_token_debug_path) if prompt_token_debug is not None else None,
        "turns": turns,
        "final_answer": final_answer,
        "total_latency_ms": total_latency_ms,
        "status": status,
        "telemetry_path": str(telemetry_path) if tegra_active else None,
        "telemetry_timeseries_path": telemetry_timeseries_path,
        "telemetry_summary": telemetry_summary,
    }


def _record_conversation_turns(
    run_paths: RunPaths,
    base_url: str,
    api_key: str | None,
    backend_fields: dict[str, Any],
    backend_max_context_tokens: int | None,
    scenario: dict[str, Any],
    context_map: dict[str, str],
    prompt_name: str,
    profile: dict[str, Any],
    thinking_enabled: bool,
    seed: int | None,
    repeat_index: int,
    with_tegrastats: bool,
) -> dict[str, Any]:
    response_base = run_paths.responses_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}"
    response_base.mkdir(parents=True, exist_ok=True)

    telemetry_path = run_paths.telemetry_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}.log"
    tegra = TegraStatsSession(telemetry_path)
    tegra_active = False
    if with_tegrastats:
        tegra_active = tegra.start()

    conversation_history: list[dict[str, Any]] = [{"role": "system", "content": _build_system_message(scenario)}]
    turns: list[dict[str, Any]] = []
    total_latency_ms = 0.0
    final_answer = ""
    status = "completed"
    session_prompt_truncated = False
    session_truncated_context_files: list[str] = []
    initial_prompt_tokens: int | None = None
    initial_prompt_token_debug: dict[str, Any] | None = None
    initial_prompt_token_debug_path: Path | None = None
    original_prompt_tokens: int | None = None
    effective_max_context_tokens: int | None = None
    scenario_declared_max_context_tokens: int | None = scenario.get("max_context_tokens")

    try:
        for turn_index, turn in enumerate(scenario["conversation_turns"], start=1):
            prompt_meta = _prepare_messages_for_conversation_turn(
                base_url=base_url,
                api_key=api_key,
                model=backend_fields["model"],
                backend_max_context_tokens=backend_max_context_tokens,
                scenario=scenario,
                context_map=context_map,
                thinking_enabled=thinking_enabled,
                generation_profile=profile,
                conversation_history=conversation_history,
                turn=turn,
                turn_index=turn_index,
            )
            if initial_prompt_tokens is None:
                initial_prompt_tokens = prompt_meta["prompt_tokens"]
                initial_prompt_token_debug = prompt_meta["prompt_token_debug"]
                initial_prompt_token_debug_path = response_base / f"turn_{turn_index:02d}_prompt_token_debug.json"
                original_prompt_tokens = prompt_meta["original_prompt_tokens"]
                effective_max_context_tokens = prompt_meta["effective_max_context_tokens"]
                scenario_declared_max_context_tokens = prompt_meta["scenario_declared_max_context_tokens"]
            session_prompt_truncated = session_prompt_truncated or prompt_meta["prompt_truncated"]
            for rel_path in prompt_meta["truncated_context_files"]:
                if rel_path not in session_truncated_context_files:
                    session_truncated_context_files.append(rel_path)

            turn_prompt_name = f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}__turn_{turn_index:02d}.json"
            _write_json(run_paths.prompts_dir / turn_prompt_name, prompt_meta["messages"])

            metrics_before = scrape_metrics(base_url, api_key)
            _write_metrics_snapshot(
                run_paths.metrics_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}__turn_{turn_index:02d}__before.prom",
                metrics_before,
            )

            completion = request_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=backend_fields["model"],
                messages=prompt_meta["messages"],
                generation=profile,
                thinking_enabled=thinking_enabled,
                prompt_token_estimate=prompt_meta["prompt_tokens"],
                seed=None if seed is None else seed + turn_index - 1,
            )

            metrics_after = scrape_metrics(base_url, api_key)
            _write_metrics_snapshot(
                run_paths.metrics_dir / f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}__turn_{turn_index:02d}__after.prom",
                metrics_after,
            )
            metric_values = metrics_delta(metrics_before, metrics_after)
            server_stats = _derive_server_stats(completion, metric_values)
            effective_ttft_ms = completion["ttft_ms"] if completion["ttft_ms"] is not None else server_stats.get("server_ttft_ms")

            events_path = response_base / f"turn_{turn_index:02d}_events.sse"
            event_timeline_path = response_base / f"turn_{turn_index:02d}_events_timeline.jsonl"
            completion_token_debug_path = response_base / f"turn_{turn_index:02d}_completion_token_fallback_debug.json"
            request_payload_path = response_base / f"turn_{turn_index:02d}_request_payload.json"
            prompt_token_debug_path = response_base / f"turn_{turn_index:02d}_prompt_token_debug.json"
            _write_text(response_base / f"turn_{turn_index:02d}_response.txt", completion["content"])
            _write_text(response_base / f"turn_{turn_index:02d}_reasoning.txt", completion["reasoning"])
            _write_json(response_base / f"turn_{turn_index:02d}_tool_calls.json", completion["tool_calls"])
            _write_text(events_path, "\n".join(completion["raw_events"]))
            _write_text(event_timeline_path, "\n".join(json.dumps(item) for item in completion["raw_event_records"]))
            _write_json(request_payload_path, completion["request_payload"])
            if prompt_meta["prompt_token_debug"] is not None:
                _write_json(prompt_token_debug_path, prompt_meta["prompt_token_debug"])
            if completion["completion_token_fallback_debug"] is not None:
                _write_json(completion_token_debug_path, completion["completion_token_fallback_debug"])

            turn_record = {
                "turn_index": turn_index,
                "task": turn["task"],
                "turn_prompt_path": str(run_paths.prompts_dir / turn_prompt_name),
                "context_files": turn.get("context_files") or scenario["context_files"],
                "response_requirements": turn.get("response_requirements") or scenario["response_requirements"],
                "prompt_token_estimate": prompt_meta["prompt_tokens"],
                "prompt_token_debug": prompt_meta["prompt_token_debug"],
                "prompt_token_debug_path": str(prompt_token_debug_path) if prompt_meta["prompt_token_debug"] is not None else None,
                "prompt_truncated": prompt_meta["prompt_truncated"],
                "latency_ms": completion["latency_ms"],
                "ttft_ms": completion["ttft_ms"],
                "ttft_ms_effective": effective_ttft_ms,
                "ttft_source": "client_stream" if completion["ttft_ms"] is not None else ("server_metrics" if effective_ttft_ms is not None else None),
                "end_to_end_tokens_per_second": completion["end_to_end_tokens_per_second"],
                "usage": completion["usage"],
                "finish_reason": completion["finish_reason"],
                "raw_text": completion["content"],
                "raw_reasoning": completion["reasoning"],
                "raw_event_count": len(completion["raw_events"]),
                "data_event_count": completion["data_event_count"],
                "content_event_count": completion["content_event_count"],
                "non_empty_text_event_count": completion["non_empty_text_event_count"],
                "reasoning_event_count": completion["reasoning_event_count"],
                "tool_call_event_count": completion["tool_call_event_count"],
                "raw_events_path": str(events_path),
                "raw_event_timeline_path": str(event_timeline_path),
                "request_payload_path": str(request_payload_path),
                "parsed_thought": completion["reasoning"],
                "parsed_answer": completion["content"],
                "parsed_tool_calls": completion["tool_calls"],
                "server_metrics_delta": metric_values,
                "server_stats": server_stats,
                "completion_token_fallback_debug": completion["completion_token_fallback_debug"],
                "completion_token_fallback_debug_path": str(completion_token_debug_path) if completion["completion_token_fallback_debug"] is not None else None,
                "reasoning_only_truncated": completion["reasoning_only_truncated"],
                "empty_stream_bug": completion["empty_stream_bug"],
            }
            turns.append(turn_record)
            total_latency_ms += completion["latency_ms"]
            final_answer = completion["content"]

            conversation_history.append({"role": "user", "content": prompt_meta["messages"][-1]["content"]})
            conversation_history.append(_assistant_message_from_completion(completion))
            _write_json(response_base / f"turn_{turn_index:02d}_continued_messages.json", conversation_history)

            if completion["reasoning_only_truncated"]:
                status = "reasoning_only_truncated"
                break

            if completion["empty_stream_bug"]:
                status = "empty_stream_bug"
                break
    finally:
        if tegra_active:
            tegra.stop()

    telemetry_summary = summarize_tegrastats_log(telemetry_path) if tegra_active else None
    telemetry_timeseries_path = write_tegrastats_timeseries(telemetry_path, tegra.interval_ms) if tegra_active else None
    return {
        "scenario_id": scenario["id"],
        "use_case_id": scenario["use_case_id"],
        "use_case_title": scenario["use_case_title"],
        "family": scenario["family"],
        "mode": scenario["mode"],
        "scenario_connectivity": scenario["scenario_connectivity"],
        "execution_mode": scenario["execution_mode"],
        "context_source": scenario["context_source"],
        "input_language": scenario.get("input_language"),
        "expected_output_language": scenario.get("expected_output_language"),
        "language_variant": scenario.get("language_variant"),
        "image_files": scenario.get("image_files") or [],
        "review_scope": scenario.get("review_scope", "single_response"),
        "variant_id": scenario.get("variant_id"),
        **backend_fields,
        "thinking_enabled": thinking_enabled,
        "repeat_index": repeat_index,
        "generation_profile": scenario["generation_profile"],
        "generation": profile,
        "seed": seed,
        "dry_run": False,
        "max_context_tokens": effective_max_context_tokens,
        "scenario_declared_max_context_tokens": scenario_declared_max_context_tokens,
        "context_budget_valid": True,
        "prompt_truncated": session_prompt_truncated,
        "original_prompt_token_estimate": original_prompt_tokens,
        "truncated_context_files": session_truncated_context_files,
        "prompt_path": str(run_paths.prompts_dir / prompt_name),
        "initial_prompt_token_estimate": initial_prompt_tokens,
        "initial_prompt_token_debug": initial_prompt_token_debug,
        "initial_prompt_token_debug_path": str(initial_prompt_token_debug_path) if initial_prompt_token_debug_path is not None and initial_prompt_token_debug is not None else None,
        "turns": turns,
        "final_answer": final_answer,
        "total_latency_ms": total_latency_ms,
        "status": status,
        "telemetry_path": str(telemetry_path) if tegra_active else None,
        "telemetry_timeseries_path": telemetry_timeseries_path,
        "telemetry_summary": telemetry_summary,
    }


def _error_record(
    scenario: dict[str, Any],
    *,
    backend_fields: dict[str, Any],
    thinking_enabled: bool,
    repeat_index: int,
    seed: int | None,
    prompt_path: str | None,
    prompt_meta: dict[str, Any] | None,
    error: Exception,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario["id"],
        "use_case_id": scenario["use_case_id"],
        "use_case_title": scenario["use_case_title"],
        "family": scenario["family"],
        "mode": scenario["mode"],
        "scenario_connectivity": scenario["scenario_connectivity"],
        "execution_mode": scenario["execution_mode"],
        "context_source": scenario["context_source"],
        "input_language": scenario.get("input_language"),
        "expected_output_language": scenario.get("expected_output_language"),
        "language_variant": scenario.get("language_variant"),
        "image_files": scenario.get("image_files") or [],
        "review_scope": scenario.get("review_scope", "single_response"),
        "variant_id": scenario.get("variant_id"),
        **backend_fields,
        "thinking_enabled": thinking_enabled,
        "repeat_index": repeat_index,
        "generation_profile": scenario["generation_profile"],
        "seed": seed,
        "dry_run": False,
        "max_context_tokens": prompt_meta.get("effective_max_context_tokens") if prompt_meta else None,
        "scenario_declared_max_context_tokens": prompt_meta.get("scenario_declared_max_context_tokens") if prompt_meta else scenario.get("max_context_tokens"),
        "context_budget_valid": prompt_meta.get("context_budget_valid") if prompt_meta else None,
        "prompt_truncated": prompt_meta.get("prompt_truncated") if prompt_meta else False,
        "original_prompt_token_estimate": prompt_meta.get("original_prompt_tokens") if prompt_meta else None,
        "truncated_context_files": prompt_meta.get("truncated_context_files") if prompt_meta else [],
        "prompt_path": prompt_path,
        "initial_prompt_token_estimate": prompt_meta.get("prompt_tokens") if prompt_meta else None,
        "initial_prompt_token_debug": prompt_meta.get("prompt_token_debug") if prompt_meta else None,
        "initial_prompt_token_debug_path": None,
        "turns": [],
        "final_answer": None,
        "total_latency_ms": None,
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
        "telemetry_path": None,
        "telemetry_timeseries_path": None,
    }


def _build_task_list(
    manifest: dict[str, Any],
    thinking_mode: str,
    families: set[str] | None,
    use_cases: set[str] | None,
    scenario_ids: set[str] | None,
    repeats: int,
    respect_repeat_count_overrides: bool,
    shuffle_seed: int | None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    think_values = [False, True] if thinking_mode == "both" else [thinking_mode == "true"]
    for scenario in manifest["scenarios"]:
        if not scenario_matches(scenario, families, use_cases):
            continue
        if scenario_ids and scenario["id"] not in scenario_ids:
            continue
        scenario_repeats = int(scenario.get("repeat_count_override", repeats)) if respect_repeat_count_overrides else repeats
        for thinking_enabled in think_values:
            for repeat_index in range(1, scenario_repeats + 1):
                tasks.append({"scenario": scenario, "thinking_enabled": thinking_enabled, "repeat_index": repeat_index})
    rng = random.Random(shuffle_seed)
    rng.shuffle(tasks)
    return tasks


def run_suite(
    project_root: Path,
    backend_config_path: Path,
    thinking_mode: str,
    output_root: Path,
    dry_run: bool = False,
    limit: int | None = None,
    families: set[str] | None = None,
    use_cases: set[str] | None = None,
    scenario_ids: set[str] | None = None,
    with_tegrastats: bool = False,
    repeats: int = 3,
    respect_repeat_count_overrides: bool = True,
    warmup_count: int = 2,
    fail_on_warmup_error: bool = True,
    shuffle_seed: int | None = 20260417,
    seed: int | None = 20260417,
) -> Path:
    backend = load_yaml(backend_config_path)
    backend_fields = _backend_record_fields(backend, backend_config_path)
    profiles = load_yaml(project_root / "configs" / "generation_profiles.yaml")
    manifest = load_yaml(project_root / "benchmarks" / "manifest.yaml")
    context_map = load_contexts(project_root)
    run_paths = ensure_run_dirs(output_root, dry_run=dry_run)

    base_url = backend["base_url"]
    api_key = backend.get("api_key")
    model = backend.get("model")
    backend_max_context_tokens = backend.get("max_context_tokens")
    if not model:
        raise ValueError(f"Backend config {backend_config_path} must pin an explicit model id.")

    model_verification = verify_model(base_url, api_key, model)
    if model_verification["advertised_models"] and not model_verification["requested_model_found"]:
        raise ValueError(
            f"Requested model {model!r} was not advertised by {base_url}. "
            f"Advertised models: {model_verification['advertised_models']}"
        )

    collect_run_metadata(
        run_paths=run_paths,
        backend=backend,
        model_verification=model_verification,
        run_options={
            "thinking_mode": thinking_mode,
            "dry_run": dry_run,
            "limit": limit,
            "families": sorted(families) if families else None,
            "use_cases": sorted(use_cases) if use_cases else None,
            "scenario_ids": sorted(scenario_ids) if scenario_ids else None,
            "with_tegrastats": with_tegrastats,
            "repeats": repeats,
            "respect_repeat_count_overrides": respect_repeat_count_overrides,
            "warmup_count": warmup_count,
            "fail_on_warmup_error": fail_on_warmup_error,
            "shuffle_seed": shuffle_seed,
            "seed": seed,
            "api_mode": "chat_completions",
        },
    )

    tasks = _build_task_list(
        manifest,
        thinking_mode,
        families,
        use_cases,
        scenario_ids,
        repeats,
        respect_repeat_count_overrides,
        shuffle_seed,
    )
    if limit is not None:
        tasks = tasks[:limit]

    warmup_tasks = tasks[: min(warmup_count, len(tasks))]
    warmup_results: list[dict[str, Any]] = []
    for warmup_index, task in enumerate(warmup_tasks, start=1):
        scenario = task["scenario"]
        if dry_run:
            continue
        try:
            if scenario["mode"] == "conversation":
                prompt_meta = _prepare_messages_for_conversation_turn(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    backend_max_context_tokens=backend_max_context_tokens,
                    scenario=scenario,
                    context_map=context_map,
                    thinking_enabled=task["thinking_enabled"],
                    generation_profile=profiles[scenario["generation_profile"]],
                    conversation_history=[{"role": "system", "content": _build_system_message(scenario)}],
                    turn=scenario["conversation_turns"][0],
                    turn_index=1,
                )
            else:
                prompt_meta = _prepare_messages_for_request(
                    project_root=project_root,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    backend_max_context_tokens=backend_max_context_tokens,
                    scenario=scenario,
                    context_map=context_map,
                    thinking_enabled=task["thinking_enabled"],
                    generation_profile=profiles[scenario["generation_profile"]],
                )
            request_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=prompt_meta["messages"],
                generation=profiles[scenario["generation_profile"]],
                thinking_enabled=task["thinking_enabled"],
                prompt_token_estimate=prompt_meta["prompt_tokens"],
                tools=scenario["tools"] if scenario["mode"] == "agent" else None,
                tool_choice="auto" if scenario["mode"] == "agent" else None,
                seed=None if seed is None else seed + 100000 + warmup_index,
            )
            warmup_results.append(
                {
                    "warmup_index": warmup_index,
                    "scenario_id": scenario["id"],
                    "thinking_enabled": task["thinking_enabled"],
                    "status": "ok",
                    "prompt_tokens": prompt_meta["prompt_tokens"],
                    "prompt_truncated": prompt_meta["prompt_truncated"],
                }
            )
        except Exception as exc:
            warmup_results.append(
                {
                    "warmup_index": warmup_index,
                    "scenario_id": scenario["id"],
                    "thinking_enabled": task["thinking_enabled"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    update_run_metadata(
        run_paths,
        lambda metadata: metadata.update(
            {
                "warmup_results": warmup_results,
                "warmup_error_count": sum(1 for item in warmup_results if item.get("status") == "error"),
            }
        ),
    )
    if not dry_run and fail_on_warmup_error:
        errors = [item for item in warmup_results if item.get("status") == "error"]
        if errors:
            first_error = errors[0]
            raise RuntimeError(
                f"Warmup failed for scenario {first_error['scenario_id']} "
                f"(thinking_enabled={first_error['thinking_enabled']}): {first_error['error_type']}: {first_error['error']}"
            )

    for task_index, task in enumerate(tasks, start=1):
        scenario = task["scenario"]
        thinking_enabled = task["thinking_enabled"]
        repeat_index = task["repeat_index"]
        profile = profiles[scenario["generation_profile"]]
        prompt_name = f"{scenario['id']}__thinking_{str(thinking_enabled).lower()}__repeat_{repeat_index:02d}.json"
        prompt_path = run_paths.prompts_dir / prompt_name
        prompt_meta: dict[str, Any] | None = None
        scenario_seed = None if seed is None else seed + task_index * 100

        try:
            if scenario["mode"] == "conversation":
                system_messages = [{"role": "system", "content": _build_system_message(scenario)}]
                _write_json(prompt_path, system_messages)
                if dry_run:
                    write_record(
                        run_paths.records_path,
                        {
                            **backend_fields,
                            "scenario_id": scenario["id"],
                            "use_case_id": scenario["use_case_id"],
                            "use_case_title": scenario["use_case_title"],
                            "family": scenario["family"],
                            "mode": scenario["mode"],
                            "scenario_connectivity": scenario["scenario_connectivity"],
                            "execution_mode": scenario["execution_mode"],
                            "context_source": scenario["context_source"],
                            "input_language": scenario.get("input_language"),
                            "expected_output_language": scenario.get("expected_output_language"),
                            "language_variant": scenario.get("language_variant"),
                            "image_files": scenario.get("image_files") or [],
                            "review_scope": scenario.get("review_scope", "single_response"),
                            "variant_id": scenario.get("variant_id"),
                            "thinking_enabled": thinking_enabled,
                            "dry_run": True,
                            "max_context_tokens": backend_max_context_tokens,
                            "scenario_declared_max_context_tokens": scenario.get("max_context_tokens"),
                            "context_budget_valid": None,
                            "prompt_truncated": False,
                            "truncated_context_files": [],
                            "repeat_index": repeat_index,
                            "prompt_path": str(prompt_path),
                            "initial_prompt_token_estimate": None,
                            "initial_prompt_token_debug": None,
                            "initial_prompt_token_debug_path": None,
                            "telemetry_path": None,
                            "telemetry_timeseries_path": None,
                        },
                    )
                    continue

                record = _record_conversation_turns(
                    run_paths=run_paths,
                    base_url=base_url,
                    api_key=api_key,
                    backend_fields=backend_fields,
                    backend_max_context_tokens=backend_max_context_tokens,
                    scenario=scenario,
                    context_map=context_map,
                    prompt_name=prompt_name,
                    profile=profile,
                    thinking_enabled=thinking_enabled,
                    seed=scenario_seed,
                    repeat_index=repeat_index,
                    with_tegrastats=with_tegrastats,
                )
                write_record(run_paths.records_path, record)
                continue

            prompt_meta = _prepare_messages_for_request(
                project_root=project_root,
                base_url=base_url,
                api_key=api_key,
                model=model,
                backend_max_context_tokens=backend_max_context_tokens,
                scenario=scenario,
                context_map=context_map,
                thinking_enabled=thinking_enabled,
                generation_profile=profile,
                estimate_tokens=not dry_run,
            )
            messages = prompt_meta["messages"]
            _write_json(prompt_path, messages)

            if dry_run:
                write_record(
                    run_paths.records_path,
                    {
                        **backend_fields,
                        "scenario_id": scenario["id"],
                        "use_case_id": scenario["use_case_id"],
                        "use_case_title": scenario["use_case_title"],
                        "family": scenario["family"],
                        "mode": scenario["mode"],
                        "scenario_connectivity": scenario["scenario_connectivity"],
                        "execution_mode": scenario["execution_mode"],
                        "context_source": scenario["context_source"],
                        "input_language": scenario.get("input_language"),
                        "expected_output_language": scenario.get("expected_output_language"),
                        "language_variant": scenario.get("language_variant"),
                        "image_files": scenario.get("image_files") or [],
                        "review_scope": scenario.get("review_scope", "single_response"),
                        "variant_id": scenario.get("variant_id"),
                        "thinking_enabled": thinking_enabled,
                        "dry_run": True,
                        "max_context_tokens": backend_max_context_tokens,
                        "scenario_declared_max_context_tokens": scenario.get("max_context_tokens"),
                        "context_budget_valid": prompt_meta["context_budget_valid"],
                        "prompt_truncated": prompt_meta["prompt_truncated"],
                        "truncated_context_files": prompt_meta["truncated_context_files"],
                        "repeat_index": repeat_index,
                        "prompt_path": str(prompt_path),
                        "initial_prompt_token_estimate": prompt_meta["prompt_tokens"],
                        "initial_prompt_token_debug": prompt_meta["prompt_token_debug"],
                        "initial_prompt_token_debug_path": None,
                        "telemetry_path": None,
                        "telemetry_timeseries_path": None,
                    },
                )
                continue

            if scenario["mode"] == "agent":
                record = _record_agent_turns(
                    run_paths=run_paths,
                    base_url=base_url,
                    api_key=api_key,
                    backend_fields=backend_fields,
                    scenario=scenario,
                    initial_messages=messages,
                    prompt_name=prompt_name,
                    profile=profile,
                    prompt_tokens=prompt_meta["prompt_tokens"],
                    prompt_token_debug=prompt_meta["prompt_token_debug"],
                    prompt_truncated=prompt_meta["prompt_truncated"],
                    original_prompt_tokens=prompt_meta["original_prompt_tokens"],
                    truncated_context_files=prompt_meta["truncated_context_files"],
                    effective_max_context_tokens=prompt_meta["effective_max_context_tokens"],
                    scenario_declared_max_context_tokens=prompt_meta["scenario_declared_max_context_tokens"],
                    context_budget_valid=prompt_meta["context_budget_valid"],
                    thinking_enabled=thinking_enabled,
                    seed=scenario_seed,
                    repeat_index=repeat_index,
                    with_tegrastats=with_tegrastats,
                )
            else:
                record = _record_non_agent_turn(
                    run_paths=run_paths,
                    base_url=base_url,
                    api_key=api_key,
                    backend_fields=backend_fields,
                    scenario=scenario,
                    messages=messages,
                    prompt_name=prompt_name,
                    profile=profile,
                    prompt_tokens=prompt_meta["prompt_tokens"],
                    prompt_token_debug=prompt_meta["prompt_token_debug"],
                    prompt_truncated=prompt_meta["prompt_truncated"],
                    original_prompt_tokens=prompt_meta["original_prompt_tokens"],
                    truncated_context_files=prompt_meta["truncated_context_files"],
                    effective_max_context_tokens=prompt_meta["effective_max_context_tokens"],
                    scenario_declared_max_context_tokens=prompt_meta["scenario_declared_max_context_tokens"],
                    context_budget_valid=prompt_meta["context_budget_valid"],
                    thinking_enabled=thinking_enabled,
                    seed=scenario_seed,
                    repeat_index=repeat_index,
                    with_tegrastats=with_tegrastats,
                )
            write_record(run_paths.records_path, record)
        except Exception as exc:
            write_record(
                run_paths.records_path,
                _error_record(
                    scenario,
                    backend_fields=backend_fields,
                    thinking_enabled=thinking_enabled,
                    repeat_index=repeat_index,
                    seed=scenario_seed,
                    prompt_path=str(prompt_path) if prompt_path.exists() else None,
                    prompt_meta=prompt_meta,
                    error=exc,
                ),
            )

    return run_paths.root
