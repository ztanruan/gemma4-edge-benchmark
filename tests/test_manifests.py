from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REQUIRED_SCENARIO_KEYS = {
    "id",
    "title",
    "use_case_id",
    "family",
    "mode",
    "generation_profile",
    "task",
}

REQUIRED_BACKEND_KEYS = {
    "name",
    "base_url",
    "model",
    "max_context_tokens",
    "supported_modalities",
    "launch_command",
}


@pytest.fixture(scope="module")
def manifest(project_root: Path) -> dict:
    return yaml.safe_load((project_root / "benchmarks" / "manifest.yaml").read_text())


def test_scenarios_have_required_keys_and_unique_ids(manifest):
    scenarios = manifest["scenarios"]
    assert scenarios
    ids = set()
    for scenario in scenarios:
        missing = REQUIRED_SCENARIO_KEYS - scenario.keys()
        assert not missing, f"scenario {scenario.get('id')} missing keys: {missing}"
        assert scenario["id"] not in ids, f"duplicate scenario id {scenario['id']}"
        ids.add(scenario["id"])


def test_generation_profiles_are_defined(manifest, project_root):
    profiles = yaml.safe_load((project_root / "configs" / "generation_profiles.yaml").read_text())
    used = {s["generation_profile"] for s in manifest["scenarios"]}
    undefined = used - profiles.keys()
    assert not undefined, f"scenarios reference undefined generation profiles: {undefined}"


def test_text_context_files_exist(manifest, project_root):
    missing: list[str] = []
    for scenario in manifest["scenarios"]:
        for rel in scenario.get("context_files") or []:
            # Image corpora are staged locally by generate_assets.py and are
            # intentionally not part of the repository.
            if rel.startswith("data/image_corpora/"):
                continue
            if not (project_root / rel).is_file():
                missing.append(f"{scenario['id']}: {rel}")
    assert not missing, f"scenarios reference missing context files: {missing[:10]}"


def test_backend_configs_are_consistent(project_root):
    backend_dir = project_root / "configs" / "backends"
    configs = sorted(backend_dir.glob("*.yaml"))
    assert configs, "no backend configs found"
    for path in configs:
        config = yaml.safe_load(path.read_text())
        missing = REQUIRED_BACKEND_KEYS - config.keys()
        assert not missing, f"{path.name} missing keys: {missing}"
        assert "text" in config["supported_modalities"]


def test_systems_manifests_parse(project_root):
    for name in ("manifest.yaml", "smoke_manifest.yaml"):
        payload = yaml.safe_load((project_root / "systems" / name).read_text())
        assert payload["experiments"], f"systems/{name} defines no experiments"
        exp_ids = [e["id"] for e in payload["experiments"]]
        assert len(exp_ids) == len(set(exp_ids)), f"duplicate experiment ids in systems/{name}"
