#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = PROJECT_ROOT / "benchmarks" / "manifest.yaml"
    if not manifest_path.exists():
        raise SystemExit("manifest.yaml not found. Run scripts/generate_assets.py first.")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    missing = []
    for scenario in manifest["scenarios"]:
        scenario_doc = PROJECT_ROOT / "docs" / "scenarios" / f"{scenario['id']}.md"
        if not scenario_doc.exists():
            missing.append(str(scenario_doc))
        for rel_path in scenario["context_files"]:
            if not (PROJECT_ROOT / rel_path).exists():
                missing.append(str(PROJECT_ROOT / rel_path))
        for rel_path in scenario.get("image_files", []):
            if not (PROJECT_ROOT / rel_path).exists():
                missing.append(str(PROJECT_ROOT / rel_path))
        if scenario.get("max_context_tokens") is None:
            missing.append(f"scenario {scenario['id']} is missing max_context_tokens")

    backends_root = PROJECT_ROOT / "configs" / "backends"
    expected_backend_paths = [
        backends_root / "vllm.yaml",
        backends_root / "vllm_baseline.yaml",
        backends_root / "vllm_image.yaml",
        backends_root / "vllm_image_280.yaml",
        backends_root / "vllm_image_560.yaml",
        backends_root / "vllm_image_1120.yaml",
        backends_root / "vllm_prefix_caching.yaml",
    ]
    for backend_path in expected_backend_paths:
        if not backend_path.exists():
            missing.append(str(backend_path))
            continue

        with backend_path.open("r", encoding="utf-8") as handle:
            backend = yaml.safe_load(handle)

        for field in (
            "model",
            "launch_command",
            "max_context_tokens",
            "max_soft_tokens",
            "supported_modalities",
            "container_image",
            "container_image_digest",
            "benchmark_profile",
            "prefix_caching_enabled",
        ):
            if backend.get(field) in (None, "", []):
                missing.append(f"{backend_path} missing required field: {field}")

        if backend.get("container_image") and "@sha256:" not in backend["container_image"]:
            missing.append(f"{backend_path} container_image is not pinned by digest")
        if not isinstance(backend.get("max_soft_tokens"), int):
            missing.append(f"{backend_path} max_soft_tokens must be an integer")
        supported_modalities = backend.get("supported_modalities") or []
        if set(supported_modalities) - {"text", "image"}:
            missing.append(f"{backend_path} supported_modalities must only contain text/image")
        if "audio" in supported_modalities or backend.get("audio_supported") not in (
            False,
            None,
        ):
            missing.append(f"{backend_path} must declare audio as unsupported for this benchmark")
        launch_command = backend.get("launch_command", "")
        if "--mm-processor-kwargs" not in launch_command:
            missing.append(f"{backend_path} launch_command must pin --mm-processor-kwargs")
        if "--limit-mm-per-prompt image=1" not in launch_command:
            missing.append(
                f"{backend_path} launch_command must cap image inputs at one per request"
            )
        if (
            bool(backend.get("prefix_caching_enabled"))
            and "--no-enable-prefix-caching" in launch_command
        ):
            missing.append(f"{backend_path} enables prefix caching but launch_command disables it")
        if (
            not bool(backend.get("prefix_caching_enabled"))
            and "--no-enable-prefix-caching" not in launch_command
        ):
            missing.append(f"{backend_path} must disable prefix caching in launch_command")

    systems_manifest_path = PROJECT_ROOT / "systems" / "manifest.yaml"
    if not systems_manifest_path.exists():
        missing.append(str(systems_manifest_path))
    else:
        with systems_manifest_path.open("r", encoding="utf-8") as handle:
            systems_manifest = yaml.safe_load(handle)
        if not systems_manifest.get("experiments"):
            missing.append(f"{systems_manifest_path} does not define any experiments")

    systems_docs_path = PROJECT_ROOT / "docs" / "systems" / "README.md"
    if not systems_docs_path.exists():
        missing.append(str(systems_docs_path))

    if missing:
        print("Missing generated files:")
        for item in missing:
            print(item)
        raise SystemExit(1)

    print(
        f"Validated {len(manifest['scenarios'])} scenarios and {len(expected_backend_paths)} backend configs successfully."
    )


if __name__ == "__main__":
    main()
