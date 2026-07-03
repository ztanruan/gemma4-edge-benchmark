from __future__ import annotations

from gemma_vllm_benchmark.catalog import USE_CASES


def test_use_case_slugs_are_unique():
    slugs = [uc.slug for uc in USE_CASES]
    assert len(slugs) == len(set(slugs))


def test_every_use_case_has_corpora_and_fixtures(project_root):
    for uc in USE_CASES:
        corpus_dir = project_root / "data" / "corpora" / uc.slug
        fixtures_dir = project_root / "data" / "tool_fixtures" / uc.slug
        assert corpus_dir.is_dir(), f"missing corpus directory for {uc.slug}"
        assert any(corpus_dir.iterdir()), f"empty corpus directory for {uc.slug}"
        for fixture in ("single_tool.json", "multi_tool.json"):
            assert (fixtures_dir / fixture).is_file(), f"missing {fixture} for {uc.slug}"


def test_every_use_case_has_a_doc(project_root):
    for uc in USE_CASES:
        doc = project_root / "docs" / "use_cases" / f"{uc.slug}.md"
        assert doc.is_file(), f"missing docs/use_cases/{uc.slug}.md"


def test_tool_templates_are_well_formed():
    for uc in USE_CASES:
        assert uc.tools, f"{uc.slug} defines no tools"
        for tool in uc.tools:
            assert tool.name
            assert tool.description
            assert tool.parameters.get("type") == "object"
