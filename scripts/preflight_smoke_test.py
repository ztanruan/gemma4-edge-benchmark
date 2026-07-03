#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.runner import (  # noqa: E402
    load_yaml,
    request_chat_completion,
    tokenize_chat_messages_detailed,
    verify_model,
)

REASONING_LEAK_PATTERNS = [
    ("gemma_channel_start", "<|channel>"),
    ("gemma_channel_end", "<channel|>"),
    ("thought_prefix", "thought\n"),
]

TOOL_LEAK_PATTERNS = [
    ("gemma_tool_decl", "<|tool>"),
    ("gemma_tool_decl_end", "<tool|>"),
    ("gemma_tool_call", "<|tool_call>"),
    ("gemma_tool_call_end", "<tool_call|>"),
    ("gemma_tool_response", "<|tool_response>"),
    ("gemma_tool_response_end", "<tool_response|>"),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _exception_payload(exc: Exception) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _persist_completion_artifacts(
    run_dir: Path, prefix: str, completion: dict[str, Any]
) -> dict[str, str]:
    paths = {
        "response_path": str(run_dir / f"{prefix}_response.txt"),
        "reasoning_path": str(run_dir / f"{prefix}_reasoning.txt"),
        "tool_calls_path": str(run_dir / f"{prefix}_tool_calls.json"),
        "events_path": str(run_dir / f"{prefix}_events.sse"),
        "timeline_path": str(run_dir / f"{prefix}_events_timeline.jsonl"),
        "payload_path": str(run_dir / f"{prefix}_request_payload.json"),
        "completion_path": str(run_dir / f"{prefix}_completion.json"),
    }
    _write_text(Path(paths["response_path"]), completion["content"])
    _write_text(Path(paths["reasoning_path"]), completion["reasoning"])
    _write_json(Path(paths["tool_calls_path"]), completion["tool_calls"])
    _write_text(Path(paths["events_path"]), "\n".join(completion["raw_events"]))
    _write_text(
        Path(paths["timeline_path"]),
        "\n".join(json.dumps(item) for item in completion["raw_event_records"]),
    )
    _write_json(Path(paths["payload_path"]), completion["request_payload"])
    _write_json(
        Path(paths["completion_path"]),
        {
            "content": completion["content"],
            "reasoning": completion["reasoning"],
            "tool_calls": completion["tool_calls"],
            "usage": completion["usage"],
            "finish_reason": completion["finish_reason"],
            "ttft_ms": completion["ttft_ms"],
            "latency_ms": completion["latency_ms"],
            "end_to_end_tokens_per_second": completion["end_to_end_tokens_per_second"],
            "data_event_count": completion["data_event_count"],
            "content_event_count": completion["content_event_count"],
            "reasoning_event_count": completion["reasoning_event_count"],
            "tool_call_event_count": completion["tool_call_event_count"],
            "reasoning_only_truncated": completion["reasoning_only_truncated"],
            "empty_stream_bug": completion["empty_stream_bug"],
        },
    )
    return paths


def _persist_generic_json(run_dir: Path, prefix: str, suffix: str, payload: Any) -> str:
    path = run_dir / f"{prefix}_{suffix}.json"
    _write_json(path, payload)
    return str(path)


def _completion_summary(completion: dict[str, Any]) -> dict[str, Any]:
    usage = completion.get("usage") or {}
    return {
        "latency_ms": completion["latency_ms"],
        "ttft_ms": completion["ttft_ms"],
        "usage": usage,
        "finish_reason": completion["finish_reason"],
        "content_chars": len(completion["content"]),
        "reasoning_chars": len(completion["reasoning"]),
        "tool_call_count": len(completion["tool_calls"]),
        "data_event_count": completion["data_event_count"],
        "content_event_count": completion["content_event_count"],
        "reasoning_event_count": completion["reasoning_event_count"],
        "tool_call_event_count": completion["tool_call_event_count"],
        "reasoning_only_truncated": completion["reasoning_only_truncated"],
        "empty_stream_bug": completion["empty_stream_bug"],
        "content_preview": completion["content"][:400],
        "reasoning_preview": completion["reasoning"][:400],
    }


def _reasoning_leak_diagnosis(completion: dict[str, Any], thinking_enabled: bool) -> dict[str, Any]:
    content = completion["content"] or ""
    markers = [name for name, pattern in REASONING_LEAK_PATTERNS if pattern in content]
    stripped = content.lstrip().lower()
    if thinking_enabled and not completion["reasoning"].strip() and stripped.startswith("thought"):
        if "thought_prefix" not in markers:
            markers.append("thought_prefix")
    detected = bool(markers)
    return {
        "detected": detected,
        "markers": markers,
        "message": (
            "Reasoning appears to have leaked into visible content instead of being separated."
            if detected
            else None
        ),
    }


def _tool_call_leak_diagnosis(text: str) -> dict[str, Any]:
    markers = [name for name, pattern in TOOL_LEAK_PATTERNS if pattern in text]
    return {
        "detected": bool(markers),
        "markers": markers,
        "message": ("Tool-call protocol tokens leaked into visible content." if markers else None),
    }


def _structured_response_format() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        "status": "HEALTHY_ONLY",
        "platform": "JETSON_THOR_ONLY",
        "checks": {
            "tokenize": "ok",
            "chat": "ok",
        },
        "count": 2,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "preflight-status",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["HEALTHY_ONLY"]},
                    "platform": {"type": "string", "enum": ["JETSON_THOR_ONLY"]},
                    "checks": {
                        "type": "object",
                        "properties": {
                            "tokenize": {"type": "string", "enum": ["ok"]},
                            "chat": {"type": "string", "enum": ["ok"]},
                        },
                        "required": ["tokenize", "chat"],
                        "additionalProperties": False,
                    },
                    "count": {"type": "integer", "enum": [2]},
                },
                "required": ["status", "platform", "checks", "count"],
                "additionalProperties": False,
            },
        },
    }
    return response_format, expected


def _validate_structured_payload(parsed: Any, expected: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return {
            "schema_valid": False,
            "exact_match": False,
            "errors": ["Top-level JSON value is not an object."],
        }
    expected_keys = set(expected.keys())
    actual_keys = set(parsed.keys())
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"Missing keys: {missing}")
    if extra:
        errors.append(f"Unexpected keys: {extra}")
    if parsed.get("status") != expected["status"]:
        errors.append("Field 'status' did not match expected enum.")
    if parsed.get("platform") != expected["platform"]:
        errors.append("Field 'platform' did not match expected enum.")
    if parsed.get("count") != expected["count"]:
        errors.append("Field 'count' did not match expected integer enum.")
    checks = parsed.get("checks")
    if not isinstance(checks, dict):
        errors.append("Field 'checks' is not an object.")
    else:
        expected_checks = expected["checks"]
        missing_checks = sorted(set(expected_checks.keys()) - set(checks.keys()))
        extra_checks = sorted(set(checks.keys()) - set(expected_checks.keys()))
        if missing_checks:
            errors.append(f"Missing nested keys in 'checks': {missing_checks}")
        if extra_checks:
            errors.append(f"Unexpected nested keys in 'checks': {extra_checks}")
        for key, value in expected_checks.items():
            if checks.get(key) != value:
                errors.append(f"Nested field 'checks.{key}' did not match expected enum.")
    return {
        "schema_valid": not errors,
        "exact_match": not errors,
        "errors": errors,
    }


def _failure_entry(
    check_name: str,
    failure_class: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check": check_name,
        "class": failure_class,
        "message": message,
        "evidence": evidence or {},
    }


def _basic_check_failure_classes(
    check_name: str,
    completion: dict[str, Any],
    *,
    thinking_enabled: bool,
    require_reasoning_and_content: bool = False,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if completion["empty_stream_bug"]:
        failures.append(
            _failure_entry(
                check_name,
                "empty_stream_bug",
                "Streamed data events arrived but no content, reasoning, or tool calls were exposed.",
                completion["empty_stream_bug"],
            )
        )
    if completion["reasoning_only_truncated"]:
        failures.append(
            _failure_entry(
                check_name,
                "reasoning_only_truncated",
                "Reasoning consumed the budget before a visible final answer was produced.",
                completion["reasoning_only_truncated"],
            )
        )
    reasoning_leak = _reasoning_leak_diagnosis(completion, thinking_enabled)
    if reasoning_leak["detected"]:
        failures.append(
            _failure_entry(check_name, "parser_leak", reasoning_leak["message"], reasoning_leak)
        )
    if require_reasoning_and_content:
        if not completion["reasoning"].strip():
            failures.append(
                _failure_entry(
                    check_name,
                    "missing_reasoning",
                    "Thinking mode was enabled but no separated reasoning field was returned.",
                )
            )
        if not completion["content"].strip():
            failures.append(
                _failure_entry(
                    check_name,
                    "missing_final_content",
                    "Thinking mode was enabled but no visible final answer content was returned.",
                )
            )
    elif not (completion["content"].strip() or completion["reasoning"].strip()):
        if not completion["tool_calls"]:
            failures.append(
                _failure_entry(
                    check_name,
                    "empty_visible_output",
                    "The completion returned neither visible content nor reasoning.",
                )
            )
    if completion["finish_reason"] == "length" and not completion["reasoning_only_truncated"]:
        failures.append(
            _failure_entry(
                check_name,
                "length_truncated",
                "The completion hit the max token limit without the reasoning-only truncation signature.",
                {"finish_reason": completion["finish_reason"]},
            )
        )
    return failures


def _run_basic_chat_check(
    *,
    run_dir: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    generation: dict[str, Any],
    thinking_enabled: bool,
    seed: int,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "You are Gemma 4 running in a preflight smoke test for vLLM on Jetson Thor. Reply concisely.",
        },
        {
            "role": "user",
            "content": "In one sentence, say that the preflight smoke test is working and mention Jetson Thor.",
        },
    ]
    tokenize_detail = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs={"enable_thinking": thinking_enabled},
        tools=None,
    )
    _write_json(run_dir / "basic_chat_tokenize.json", tokenize_detail)
    if not (tokenize_detail.get("ok") and isinstance(tokenize_detail.get("count"), int)):
        return {
            "status": "error",
            "tokenize": tokenize_detail,
            "failure_classes": [
                _failure_entry(
                    "basic_chat_completion",
                    "tokenize_failure",
                    "Chat-aware /tokenize failed for the basic chat preflight check.",
                    tokenize_detail,
                )
            ],
        }

    completion = request_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        generation=generation,
        thinking_enabled=thinking_enabled,
        prompt_token_estimate=int(tokenize_detail["count"]),
        seed=seed,
    )
    artifacts = _persist_completion_artifacts(run_dir, "basic", completion)
    failures = _basic_check_failure_classes(
        "basic_chat_completion",
        completion,
        thinking_enabled=thinking_enabled,
        require_reasoning_and_content=thinking_enabled,
    )
    diagnosis = {
        **_completion_summary(completion),
        "reasoning_leak": _reasoning_leak_diagnosis(completion, thinking_enabled),
        "failure_classes": failures,
        "human_review_focus": [
            "Whether reasoning is separated into the reasoning field.",
            "Whether final visible answer content exists.",
            "Whether truncation is genuine or caused by parser leakage.",
        ],
    }
    diagnosis_path = _persist_generic_json(run_dir, "basic", "diagnosis", diagnosis)
    return {
        "status": "ok" if not failures else "error",
        "tokenize": tokenize_detail,
        **diagnosis,
        "artifacts": {**artifacts, "diagnosis_path": diagnosis_path},
    }


def _run_structured_output_check(
    *,
    run_dir: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    generation: dict[str, Any],
    thinking_enabled: bool,
    seed: int,
) -> dict[str, Any]:
    response_format, expected_payload = _structured_response_format()
    messages = [
        {
            "role": "system",
            "content": (
                "You are Gemma 4 in a structured-output smoke test. "
                "Return concise structured JSON that follows the requested schema exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return a JSON object with exactly these values: "
                "status=HEALTHY_ONLY, platform=JETSON_THOR_ONLY, "
                "checks.tokenize=ok, checks.chat=ok, count=2."
            ),
        },
    ]
    tokenize_detail = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs={"enable_thinking": thinking_enabled},
        tools=None,
    )
    _write_json(run_dir / "structured_chat_tokenize.json", tokenize_detail)
    if not (tokenize_detail.get("ok") and isinstance(tokenize_detail.get("count"), int)):
        return {
            "status": "error",
            "tokenize": tokenize_detail,
            "failure_classes": [
                _failure_entry(
                    "structured_output_chat_completion",
                    "tokenize_failure",
                    "Chat-aware /tokenize failed for the structured-output preflight check.",
                    tokenize_detail,
                )
            ],
        }

    completion = request_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        generation=generation,
        thinking_enabled=thinking_enabled,
        prompt_token_estimate=int(tokenize_detail["count"]),
        response_format=response_format,
        seed=seed,
    )
    artifacts = _persist_completion_artifacts(run_dir, "structured", completion)
    parsed_json = None
    parse_error = None
    try:
        parsed_json = json.loads(completion["content"])
    except ValueError as exc:
        parse_error = str(exc)
    validation = (
        _validate_structured_payload(parsed_json, expected_payload)
        if parse_error is None
        else {
            "schema_valid": False,
            "exact_match": False,
            "errors": [f"Invalid JSON: {parse_error}"],
        }
    )
    failures = _basic_check_failure_classes(
        "structured_output_chat_completion",
        completion,
        thinking_enabled=thinking_enabled,
        require_reasoning_and_content=False,
    )
    if not validation["schema_valid"]:
        failures.append(
            _failure_entry(
                "structured_output_chat_completion",
                "structured_output_bypass",
                "Structured output did not satisfy the constrained schema.",
                validation,
            )
        )
    diagnosis = {
        **_completion_summary(completion),
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "expected_payload": expected_payload,
        "validation": validation,
        "reasoning_leak": _reasoning_leak_diagnosis(completion, thinking_enabled),
        "failure_classes": failures,
        "human_review_focus": [
            "Whether the JSON shape and exact enum values were enforced.",
            "Whether reasoning leaked into content.",
            "Whether invalid JSON or missing fields indicate structured-output bypass.",
        ],
    }
    diagnosis_path = _persist_generic_json(run_dir, "structured", "diagnosis", diagnosis)
    return {
        "status": "ok" if not failures else "error",
        "thinking_enabled": thinking_enabled,
        "tokenize": tokenize_detail,
        **diagnosis,
        "artifacts": {**artifacts, "diagnosis_path": diagnosis_path},
    }


def _run_tool_check(
    *,
    run_dir: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    generation: dict[str, Any],
    thinking_enabled: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_site_status",
                "description": "Returns the local site health status for a named edge site.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site": {
                            "type": "string",
                            "description": "Short edge site identifier",
                        }
                    },
                    "required": ["site"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are Gemma 4 in a tool-calling smoke test. Use the provided tool when the user explicitly asks for site status.",
        },
        {
            "role": "user",
            "content": "Use the available tool to check the status for site alpha and do not answer from prior knowledge.",
        },
    ]
    tokenize_detail = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs={"enable_thinking": thinking_enabled},
        tools=tools,
    )
    _write_json(run_dir / "tool_chat_tokenize.json", tokenize_detail)
    if not (tokenize_detail.get("ok") and isinstance(tokenize_detail.get("count"), int)):
        result = {
            "status": "error",
            "tokenize": tokenize_detail,
            "failure_classes": [
                _failure_entry(
                    "tool_chat_completion",
                    "tokenize_failure",
                    "Tool-aware /tokenize failed for the tool preflight check.",
                    tokenize_detail,
                )
            ],
        }
        return result, result

    completion = request_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        generation=generation,
        thinking_enabled=thinking_enabled,
        prompt_token_estimate=int(tokenize_detail["count"]),
        tools=tools,
        tool_choice="auto",
        seed=seed,
    )
    artifacts = _persist_completion_artifacts(run_dir, "tool", completion)
    failures = _basic_check_failure_classes(
        "tool_chat_completion",
        completion,
        thinking_enabled=thinking_enabled,
        require_reasoning_and_content=False,
    )
    tool_leak = _tool_call_leak_diagnosis(completion["content"])
    if tool_leak["detected"]:
        failures.append(
            _failure_entry(
                "tool_chat_completion",
                "tool_call_leak",
                tool_leak["message"],
                tool_leak,
            )
        )
    if len(completion["tool_calls"]) == 0:
        failures.append(
            _failure_entry(
                "tool_chat_completion",
                "missing_tool_call",
                "Tool chat completion did not emit any parsed tool calls.",
            )
        )
    diagnosis = {
        **_completion_summary(completion),
        "parsed_tool_calls": completion["tool_calls"],
        "tool_call_leak": tool_leak,
        "reasoning_leak": _reasoning_leak_diagnosis(completion, thinking_enabled),
        "failure_classes": failures,
        "human_review_focus": [
            "Whether parsed tool_calls are emitted instead of raw protocol tokens in content.",
            "Whether reasoning leaks into content.",
            "Whether the request stops for a tool call cleanly.",
        ],
    }
    diagnosis_path = _persist_generic_json(run_dir, "tool", "diagnosis", diagnosis)
    first_result = {
        "status": "ok" if not failures else "error",
        "tokenize": tokenize_detail,
        **diagnosis,
        "artifacts": {**artifacts, "diagnosis_path": diagnosis_path},
    }

    if not completion["tool_calls"]:
        follow_up_result = {
            "status": "skipped",
            "reason": "no_tool_call_from_first_step",
            "failure_classes": [],
        }
        return first_result, follow_up_result

    first_tool_call = completion["tool_calls"][0]
    follow_up_messages = [
        messages[0],
        {
            "role": "assistant",
            "content": completion["content"] or None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": call["type"],
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    },
                }
                for call in completion["tool_calls"]
            ],
        },
        {
            "role": "tool",
            "tool_call_id": first_tool_call["id"],
            "content": json.dumps(
                {
                    "site": "alpha",
                    "status": "healthy",
                    "last_checked": "2026-04-17T09:00:00Z",
                }
            ),
        },
    ]
    follow_up_tokenize = tokenize_chat_messages_detailed(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=follow_up_messages,
        add_special_tokens=True,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs={"enable_thinking": thinking_enabled},
        tools=tools,
    )
    _write_json(run_dir / "tool_follow_up_chat_tokenize.json", follow_up_tokenize)
    if not (follow_up_tokenize.get("ok") and isinstance(follow_up_tokenize.get("count"), int)):
        follow_up_result = {
            "status": "error",
            "tokenize": follow_up_tokenize,
            "failure_classes": [
                _failure_entry(
                    "tool_follow_up_chat_completion",
                    "tokenize_failure",
                    "Tool follow-up /tokenize failed.",
                    follow_up_tokenize,
                )
            ],
        }
        return first_result, follow_up_result

    follow_up_completion = request_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=follow_up_messages,
        generation=generation,
        thinking_enabled=thinking_enabled,
        prompt_token_estimate=int(follow_up_tokenize["count"]),
        tools=tools,
        seed=seed + 10,
    )
    follow_up_artifacts = _persist_completion_artifacts(
        run_dir, "tool_follow_up", follow_up_completion
    )
    follow_up_failures = _basic_check_failure_classes(
        "tool_follow_up_chat_completion",
        follow_up_completion,
        thinking_enabled=thinking_enabled,
        require_reasoning_and_content=False,
    )
    follow_up_leak = _tool_call_leak_diagnosis(follow_up_completion["content"])
    if follow_up_leak["detected"]:
        follow_up_failures.append(
            _failure_entry(
                "tool_follow_up_chat_completion",
                "tool_call_leak",
                follow_up_leak["message"],
                follow_up_leak,
            )
        )
    if not (follow_up_completion["content"].strip() or follow_up_completion["reasoning"].strip()):
        follow_up_failures.append(
            _failure_entry(
                "tool_follow_up_chat_completion",
                "empty_visible_output",
                "Tool follow-up completion returned neither visible content nor reasoning.",
            )
        )
    diagnosis = {
        **_completion_summary(follow_up_completion),
        "tool_call_leak": follow_up_leak,
        "reasoning_leak": _reasoning_leak_diagnosis(follow_up_completion, thinking_enabled),
        "failure_classes": follow_up_failures,
        "human_review_focus": [
            "Whether the follow-up answer is visible content instead of raw tool-call syntax.",
            "Whether reasoning leaks into content.",
            "Whether the tool result is consumed cleanly in the final answer step.",
        ],
    }
    diagnosis_path = _persist_generic_json(run_dir, "tool_follow_up", "diagnosis", diagnosis)
    follow_up_result = {
        "status": "ok" if not follow_up_failures else "error",
        "tokenize": follow_up_tokenize,
        **diagnosis,
        "artifacts": {**follow_up_artifacts, "diagnosis_path": diagnosis_path},
    }
    return first_result, follow_up_result


def _run_image_check(
    *,
    run_dir: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    generation: dict[str, Any],
    thinking_enabled: bool,
    seed: int,
) -> dict[str, Any]:
    sample_images = sorted((PROJECT_ROOT / "data" / "image_corpora").rglob("*"))
    sample_image = next(
        (
            path
            for path in sample_images
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        None,
    )
    if sample_image is None:
        return {
            "status": "skipped",
            "reason": "no_staged_local_image_found",
            "failure_classes": [],
        }

    rel_image = sample_image.relative_to(PROJECT_ROOT).as_posix()
    messages = [
        {
            "role": "system",
            "content": "You are Gemma 4 in an image-classification smoke test. Choose one allowed label and answer concisely.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{sample_image.resolve()}"},
                    "uuid": rel_image,
                },
                {
                    "type": "text",
                    "text": "Classify the image using exactly one label from this closed set: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Return JSON only.",
                },
            ],
        },
    ]
    completion = request_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        generation=generation,
        thinking_enabled=thinking_enabled,
        prompt_token_estimate=None,
        seed=seed,
    )
    artifacts = _persist_completion_artifacts(run_dir, "image", completion)
    failures = _basic_check_failure_classes(
        "image_chat_completion",
        completion,
        thinking_enabled=thinking_enabled,
        require_reasoning_and_content=False,
    )
    diagnosis = {
        **_completion_summary(completion),
        "sample_image": rel_image,
        "reasoning_leak": _reasoning_leak_diagnosis(completion, thinking_enabled),
        "failure_classes": failures,
        "human_review_focus": [
            "Whether multimodal content is processed and visible output exists.",
            "Whether reasoning leaks into content.",
            "Whether the image path and chat payload are accepted cleanly.",
        ],
    }
    diagnosis_path = _persist_generic_json(run_dir, "image", "diagnosis", diagnosis)
    return {
        "status": "ok" if not failures else "error",
        **diagnosis,
        "artifacts": {
            **artifacts,
            "diagnosis_path": diagnosis_path,
            "image_path": str(sample_image),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "preflight")
    parser.add_argument("--thinking", choices=["true", "false"], default="false")
    parser.add_argument("--skip-tool-check", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--seed", type=int, default=20260417)
    args = parser.parse_args()

    backend = load_yaml(args.backend_config)
    base_url = backend["base_url"]
    api_key = backend.get("api_key")
    model = backend.get("model")
    if not model:
        raise SystemExit(f"Backend config {args.backend_config} must pin an explicit model id.")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    thinking_enabled = args.thinking == "true"
    generation = {
        "max_tokens": 2096 if thinking_enabled else 512,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
    }

    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend_config": str(args.backend_config),
        "base_url": base_url,
        "model": model,
        "backend_profile": backend.get("benchmark_profile"),
        "supported_modalities": backend.get("supported_modalities"),
        "max_soft_tokens": backend.get("max_soft_tokens"),
        "thinking_enabled": thinking_enabled,
        "generation": generation,
        "checks": {},
        "failures": [],
    }

    model_verification = verify_model(base_url, api_key, model)
    report["checks"]["model_verification"] = model_verification
    if model_verification["advertised_models"] and not model_verification["requested_model_found"]:
        report["failures"].append(
            _failure_entry(
                "model_verification",
                "model_not_advertised",
                f"Requested model {model!r} was not advertised by {base_url}.",
                model_verification,
            )
        )

    try:
        report["checks"]["basic_chat_completion"] = _run_basic_chat_check(
            run_dir=run_dir,
            base_url=base_url,
            api_key=api_key,
            model=model,
            generation=generation,
            thinking_enabled=thinking_enabled,
            seed=args.seed,
        )
        report["failures"].extend(
            report["checks"]["basic_chat_completion"].get("failure_classes", [])
        )
    except Exception as exc:
        report["checks"]["basic_chat_completion"] = {
            "status": "error",
            **_exception_payload(exc),
        }
        report["failures"].append(
            _failure_entry(
                "basic_chat_completion",
                "exception",
                "Basic chat preflight check raised an exception.",
                report["checks"]["basic_chat_completion"],
            )
        )

    try:
        report["checks"]["structured_output_chat_completion"] = _run_structured_output_check(
            run_dir=run_dir,
            base_url=base_url,
            api_key=api_key,
            model=model,
            generation=generation,
            thinking_enabled=thinking_enabled,
            seed=args.seed + 20,
        )
        report["failures"].extend(
            report["checks"]["structured_output_chat_completion"].get("failure_classes", [])
        )
    except Exception as exc:
        report["checks"]["structured_output_chat_completion"] = {
            "status": "error",
            **_exception_payload(exc),
        }
        report["failures"].append(
            _failure_entry(
                "structured_output_chat_completion",
                "exception",
                "Structured-output preflight check raised an exception.",
                report["checks"]["structured_output_chat_completion"],
            )
        )

    if args.skip_tool_check:
        report["checks"]["tool_chat_completion"] = {
            "status": "skipped",
            "reason": "skip_tool_check",
            "failure_classes": [],
        }
        report["checks"]["tool_follow_up_chat_completion"] = {
            "status": "skipped",
            "reason": "skip_tool_check",
            "failure_classes": [],
        }
    else:
        try:
            first_result, follow_up_result = _run_tool_check(
                run_dir=run_dir,
                base_url=base_url,
                api_key=api_key,
                model=model,
                generation=generation,
                thinking_enabled=thinking_enabled,
                seed=args.seed + 1,
            )
            report["checks"]["tool_chat_completion"] = first_result
            report["checks"]["tool_follow_up_chat_completion"] = follow_up_result
            report["failures"].extend(first_result.get("failure_classes", []))
            report["failures"].extend(follow_up_result.get("failure_classes", []))
        except Exception as exc:
            payload = {"status": "error", **_exception_payload(exc)}
            report["checks"]["tool_chat_completion"] = payload
            report["checks"]["tool_follow_up_chat_completion"] = {
                "status": "skipped",
                "reason": "tool_check_exception",
                "failure_classes": [],
            }
            report["failures"].append(
                _failure_entry(
                    "tool_chat_completion",
                    "exception",
                    "Tool-calling preflight check raised an exception.",
                    payload,
                )
            )

    if args.skip_image_check:
        report["checks"]["image_chat_completion"] = {
            "status": "skipped",
            "reason": "skip_image_check",
            "failure_classes": [],
        }
    else:
        try:
            report["checks"]["image_chat_completion"] = _run_image_check(
                run_dir=run_dir,
                base_url=base_url,
                api_key=api_key,
                model=model,
                generation=generation,
                thinking_enabled=thinking_enabled,
                seed=args.seed + 2,
            )
            report["failures"].extend(
                report["checks"]["image_chat_completion"].get("failure_classes", [])
            )
        except Exception as exc:
            report["checks"]["image_chat_completion"] = {
                "status": "error",
                **_exception_payload(exc),
            }
            report["failures"].append(
                _failure_entry(
                    "image_chat_completion",
                    "exception",
                    "Image preflight check raised an exception.",
                    report["checks"]["image_chat_completion"],
                )
            )

    report["status"] = "ok" if not report["failures"] else "error"
    report["failure_count"] = len(report["failures"])
    report_path = run_dir / "preflight_report.json"
    _write_json(report_path, report)
    print(run_dir)
    print(report_path)
    if report["status"] != "ok":
        raise SystemExit(f"Preflight failed with {report['failure_count']} classified issue(s).")


if __name__ == "__main__":
    main()
