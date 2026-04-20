from __future__ import annotations

import csv
import json
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Any

from PIL import Image


CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CALTECH256_SELECTED_CLASSES = {
    "003.backpack": "backpack",
    "004.baseball-bat": "baseball bat",
    "008.bathtub": "bathtub",
    "009.bear": "bear",
    "010.beer-mug": "beer mug",
    "025.cactus": "cactus",
    "028.camel": "camel",
    "045.computer-keyboard": "computer keyboard",
    "056.dog": "dog",
    "084.giraffe": "giraffe",
}

MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES = [
    "en",
    "id",
    "ja",
    "kk",
    "ko",
    "ru",
    "ur",
    "uz",
    "vi",
    "zh-cn",
    "zh-tw",
]

IMAGE_TIER_SAMPLES_PER_CLASS = {
    "small": 20,
    "medium": 50,
    "large": 100,
}

CLOCK_TIME_TIER_SAMPLES_PER_LABEL = {
    "small": 2,
    "medium": 7,
    "large": 10,
}

CLOCK_TIME_LABELS = [f"{hour}_{minute:02d}" for hour in range(1, 13) for minute in range(0, 60, 5)]

DEFAULT_IMAGE_TIER = "medium"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_use_case_doc(title: str, summary: str, rationale: str) -> str:
    return f"""# {title}

## Summary

{summary}

## Why This Matters on Jetson

{rationale}

## Benchmark Semantics

- This is a closed-set image classification benchmark.
- The benchmark sends local images through vLLM's `/v1/chat/completions` API using OpenAI-style multimodal `messages`.
- The image comes first in the user content array, then the text instruction, following Gemma multimodal best practice.
- The benchmark records outputs and metrics only. It does not score correctness automatically.
"""


def _write_multilingual_ocr_use_case_doc(
    *,
    summary: str,
    rationale: str,
    language_codes: list[str],
    available_counts: dict[str, int],
    staged_per_language: int,
) -> str:
    counts_lines = "\n".join(f"- `{code}`: {available_counts.get(code, 0)} available" for code in language_codes)
    return f"""# Multilingual Image Text Extraction

## Summary

{summary}

## Why This Matters on Jetson

{rationale}

## Languages

This staged OCR benchmark includes all languages present in the local source copy:

{counts_lines}

Current staged sample count per language: `{staged_per_language}`

## Benchmark Semantics

- This is a multilingual OCR-style extraction benchmark, not image classification.
- The benchmark sends local images through vLLM's `/v1/chat/completions` API using OpenAI-style multimodal `messages`.
- The image comes first in the user content array, then the text instruction, following Gemma multimodal best practice.
- The model is asked to return JSON only with `extracted_text`, `detected_language`, and `confidence_band`.
- The benchmark records outputs and metrics only. It does not score correctness automatically.
"""


def _samples_per_class_for_tier(image_tier: str) -> int:
    try:
        return IMAGE_TIER_SAMPLES_PER_CLASS[image_tier]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported image tier {image_tier!r}. Choose from: {', '.join(sorted(IMAGE_TIER_SAMPLES_PER_CLASS))}."
        ) from exc


def _write_clock_time_use_case_doc(
    *,
    summary: str,
    rationale: str,
    samples_per_label: int,
    total_images: int,
) -> str:
    return f"""# Analog Clock Time Reading

## Summary

{summary}

## Why This Matters on Jetson

{rationale}

## Label Space

- Closed-set time labels: `{len(CLOCK_TIME_LABELS)}`
- Label format: `H_MM`
- Example: `5_40` means `5:40`
- Staged split: held-out `test`
- Current staged sample count per time label: `{samples_per_label}`
- Total staged images: `{total_images}`

## Benchmark Semantics

- This is a closed-set analog clock reading benchmark, not free-form captioning.
- The benchmark sends local images through vLLM's `/v1/chat/completions` API using OpenAI-style multimodal `messages`.
- The image comes first in the user content array, then the text instruction, following Gemma multimodal best practice.
- The model is asked to return JSON only with `predicted_label`, `confidence_band`, and `brief_reason`.
- `predicted_label` must be exactly one of the allowed `H_MM` time labels.
- The benchmark records outputs and metrics only. It does not score correctness automatically.
"""


def _clock_time_samples_per_label_for_tier(image_tier: str) -> int:
    try:
        return CLOCK_TIME_TIER_SAMPLES_PER_LABEL[image_tier]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported image tier {image_tier!r}. Choose from: {', '.join(sorted(CLOCK_TIME_TIER_SAMPLES_PER_LABEL))}."
        ) from exc


def _preferred_dataset_dir(project_root: Path, dirname: str, archive_name: str) -> tuple[Path | None, list[Path]]:
    candidates = [
        project_root / "data" / "source_datasets" / dirname,
        Path.home() / "Downloads" / dirname,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    archive_candidates = [
        project_root / "data" / "source_datasets" / archive_name,
        Path.home() / "Downloads" / archive_name,
    ]
    return None, candidates + archive_candidates


def _load_cifar_batch(path: Path) -> dict[bytes, Any]:
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            )
            return pickle.load(handle, encoding="bytes")


def _render_cifar_png(image_bytes: list[int], output_path: Path, upscale_size: int = 256) -> None:
    image = Image.new("RGB", (32, 32))
    pixels = list(image_bytes)
    reds = pixels[0:1024]
    greens = pixels[1024:2048]
    blues = pixels[2048:3072]
    rgb_pixels = []
    for idx in range(1024):
        rgb_pixels.append((reds[idx], greens[idx], blues[idx]))
    image.putdata(rgb_pixels)
    image = image.resize((upscale_size, upscale_size), Image.Resampling.BICUBIC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _render_caltech_jpeg(input_path: Path, output_path: Path, max_side: int = 512) -> None:
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=92)


def _candidate_list_text(labels: list[str]) -> str:
    return ", ".join(labels)


def _time_label_to_display(label: str) -> str:
    hour, minute = label.split("_", maxsplit=1)
    return f"{int(hour)}:{minute}"


def _image_classification_scenario(
    *,
    scenario_id: str,
    title: str,
    use_case_id: str,
    use_case_title: str,
    family: str,
    image_rel_path: str,
    correct_label: str,
    candidate_labels: list[str],
    dataset_name: str,
    source_reference: str,
) -> dict[str, Any]:
    description = (
        f"Tests closed-set image classification on a staged {dataset_name} sample using an explicit allowed label set."
    )
    return {
        "id": scenario_id,
        "title": title,
        "description": description,
        "use_case_id": use_case_id,
        "use_case_title": use_case_title,
        "family": family,
        "mode": "non_agent",
        "scenario_connectivity": "offline",
        "execution_mode": "mocked",
        "context_source": "local_image_dataset",
        "review_scope": "single_response",
        "input_modality": "image",
        "skip_prompt_budget_enforcement": True,
        "max_context_tokens": 65536,
        "context_files": [],
        "image_files": [image_rel_path],
        "task": (
            "Classify the image using exactly one label from this closed set: "
            f"{_candidate_list_text(candidate_labels)}. "
            "Return JSON only with keys predicted_label, confidence_band, and brief_reason. "
            "The predicted_label must be one of the allowed labels exactly. "
            "The confidence_band must be one of high, medium, or low. "
            "Keep brief_reason under 20 words and grounded only in visible image evidence. "
            "If uncertain, still choose the single best label from the allowed set."
        ),
        "response_requirements": [
            "Return JSON only.",
            "Use exactly one label from the allowed set.",
            "Use confidence_band of high, medium, or low.",
            "Do not invent labels outside the allowed set.",
            "Keep brief_reason short and image-grounded.",
        ],
        "generation_profile": "gemma_image_classification",
        "tools": [],
        "tool_results": [],
        "expected_tool_calls": [],
        "judge": {
            "reference_answer": [
                f"Correct dataset label is {correct_label}.",
                f"Allowed label set is: {_candidate_list_text(candidate_labels)}.",
                f"Source reference: {source_reference}.",
            ],
            "must_include": [
                "predicted_label uses exactly one allowed label.",
                "brief_reason is grounded in the visible image rather than invented metadata.",
                f"The later reviewer should compare predicted_label against the reference label {correct_label}.",
            ],
            "should_avoid": [
                "Labels outside the allowed set.",
                "Long prose instead of JSON.",
                "Referring to dataset metadata that was not shown in the prompt.",
            ],
            "judge_questions": [
                "Is the output valid JSON with the required keys?",
                "Does predicted_label stay within the allowed label set?",
                f"Does the predicted_label match the reference label `{correct_label}`?",
            ],
        },
        "image_dataset": dataset_name,
        "image_label": correct_label,
    }


def _multilingual_image_text_extraction_scenario(
    *,
    scenario_id: str,
    title: str,
    image_rel_path: str,
    sample_id: str,
    reference_text: str,
    language_code: str,
    language_name: str,
    script: str,
) -> dict[str, Any]:
    return {
        "id": scenario_id,
        "title": title,
        "description": (
            "Tests whether the model can read visible text from a multilingual image accurately, keep the text in the source language, "
            "and return a structured extraction payload."
        ),
        "use_case_id": "image_text_extraction_multilingual",
        "use_case_title": "Multilingual Image Text Extraction",
        "family": "image_text_extraction_multilingual",
        "mode": "non_agent",
        "scenario_connectivity": "offline",
        "execution_mode": "mocked",
        "context_source": "local_image_dataset",
        "review_scope": "single_response",
        "input_modality": "image",
        "input_language": language_name,
        "expected_output_language": "JSON with source-language transcript",
        "language_variant": "same_language_ocr",
        "skip_prompt_budget_enforcement": True,
        "max_context_tokens": 65536,
        "context_files": [],
        "image_files": [image_rel_path],
        "task": (
            "Read the text visible in the image and return JSON only with keys extracted_text, detected_language, and confidence_band. "
            "Do not translate, summarize, or explain. "
            f"The detected_language must be exactly one of: {', '.join(MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES)}. "
            "Preserve wording and punctuation as faithfully as possible. "
            "The confidence_band must be one of high, medium, or low."
        ),
        "response_requirements": [
            "Return JSON only.",
            "Do not translate the visible text into another language.",
            "Use detected_language from the allowed code set only.",
            "Keep extracted_text faithful to the image content.",
            "Use confidence_band of high, medium, or low.",
        ],
        "generation_profile": "gemma_image_text_extraction",
        "tools": [],
        "tool_results": [],
        "expected_tool_calls": [],
        "judge": {
            "reference_answer": [
                f"Reference text: {reference_text}",
                f"Reference detected_language: {language_code}",
                f"Reference script: {script}",
                f"Source sample: {sample_id}",
            ],
            "must_include": [
                "Valid JSON with extracted_text, detected_language, and confidence_band.",
                "The transcript should preserve the visible source-language content rather than translating it.",
                f"The later reviewer should compare extracted_text against the reference text and detected_language against `{language_code}`.",
            ],
            "should_avoid": [
                "Translating the text instead of transcribing it.",
                "Inventing words not present in the image.",
                "Returning prose or markdown instead of JSON.",
            ],
            "judge_questions": [
                "Is the output valid JSON with the required keys?",
                "Does extracted_text faithfully match the image text, allowing only minor normalization differences if the wording is preserved?",
                f"Is detected_language correctly set to `{language_code}`?",
            ],
        },
        "image_dataset": "multilingual-image-text-translation",
        "image_label": language_code,
        "expected_extracted_text": reference_text,
        "expected_language_code": language_code,
    }


def _clock_time_reading_scenario(
    *,
    scenario_id: str,
    title: str,
    image_rel_path: str,
    correct_label: str,
    source_reference: str,
) -> dict[str, Any]:
    display_time = _time_label_to_display(correct_label)
    return {
        "id": scenario_id,
        "title": title,
        "description": (
            "Tests whether the model can read an analog clock face and choose the exact time from the closed-set label space."
        ),
        "use_case_id": "image_clock_time_reading",
        "use_case_title": "Analog Clock Time Reading",
        "family": "image_clock_time_reading",
        "mode": "non_agent",
        "scenario_connectivity": "offline",
        "execution_mode": "mocked",
        "context_source": "local_image_dataset",
        "review_scope": "single_response",
        "input_modality": "image",
        "skip_prompt_budget_enforcement": True,
        "max_context_tokens": 65536,
        "context_files": [],
        "image_files": [image_rel_path],
        "task": (
            "Read the analog clock and classify the time using exactly one label from this closed set: "
            f"{_candidate_list_text(CLOCK_TIME_LABELS)}. "
            "Each label uses the format H_MM where the underscore separates hours and minutes, "
            "for example 5_40 means 5:40 and 12_05 means 12:05. "
            "Return JSON only with keys predicted_label, confidence_band, and brief_reason. "
            "The predicted_label must be one of the allowed labels exactly. "
            "The confidence_band must be one of high, medium, or low. "
            "Keep brief_reason under 20 words and grounded only in visible clock-hand evidence. "
            "If uncertain, still choose the single best label from the allowed set."
        ),
        "response_requirements": [
            "Return JSON only.",
            "Use exactly one time label from the allowed set.",
            "Use confidence_band of high, medium, or low.",
            "Do not invent labels outside the allowed set.",
            "Keep brief_reason short and grounded in the visible clock hands.",
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "clock-time-reading",
                "schema": {
                    "type": "object",
                    "properties": {
                        "predicted_label": {"type": "string", "enum": CLOCK_TIME_LABELS},
                        "confidence_band": {"type": "string", "enum": ["high", "medium", "low"]},
                        "brief_reason": {"type": "string"},
                    },
                    "required": ["predicted_label", "confidence_band", "brief_reason"],
                    "additionalProperties": False,
                },
            },
        },
        "generation_profile": "gemma_clock_time_reading",
        "tools": [],
        "tool_results": [],
        "expected_tool_calls": [],
        "judge": {
            "reference_answer": [
                f"Correct time label is {correct_label}.",
                f"Correct human-readable time is {display_time}.",
                "Labels are exact closed-set choices in H_MM format.",
                f"Source reference: {source_reference}.",
            ],
            "must_include": [
                "predicted_label uses exactly one allowed H_MM label.",
                "brief_reason is grounded in visible clock-hand positions rather than invented metadata.",
                f"The later reviewer should compare predicted_label against the reference label {correct_label} ({display_time}).",
            ],
            "should_avoid": [
                "Labels outside the allowed set.",
                "Returning prose or markdown instead of JSON.",
                "Guessing a nearby time while ignoring the exact minute label.",
            ],
            "judge_questions": [
                "Is the output valid JSON with the required keys?",
                "Does predicted_label stay within the allowed H_MM label set?",
                f"Does the predicted_label exactly match the reference label `{correct_label}`?",
            ],
        },
        "image_dataset": "time-image-datasetclassification",
        "image_label": correct_label,
        "expected_time_label": correct_label,
        "expected_time_text": display_time,
    }


def stage_image_benchmarks(project_root: Path, image_tier: str = DEFAULT_IMAGE_TIER) -> dict[str, Any]:
    samples_per_class = _samples_per_class_for_tier(image_tier)
    clock_samples_per_label = _clock_time_samples_per_label_for_tier(image_tier)
    image_root = project_root / "data" / "image_corpora"
    cifar_root = image_root / "cifar10_subset"
    caltech_root = image_root / "caltech256_subset"
    multilingual_root = image_root / "multilingual_image_text_subset"
    clock_root = image_root / "clock_time_subset"
    _reset_dir(cifar_root)
    _reset_dir(caltech_root)
    _reset_dir(multilingual_root)
    _reset_dir(clock_root)

    cifar_dir, cifar_candidates = _preferred_dataset_dir(project_root, "cifar-10-batches-py", "cifar-10-python.tar.gz")
    if cifar_dir is None:
        raise FileNotFoundError(
            "Missing CIFAR-10 source directory. Checked: "
            + ", ".join(str(path) for path in cifar_candidates)
            + ". Place the extracted CIFAR-10 Python dataset under "
            + f"{project_root / 'data' / 'source_datasets' / 'cifar-10-batches-py'}."
        )
    caltech_dir, caltech_candidates = _preferred_dataset_dir(project_root, "256_ObjectCategories", "256_ObjectCategories.tar")
    if caltech_dir is None:
        raise FileNotFoundError(
            "Missing Caltech 256 source directory. Checked: "
            + ", ".join(str(path) for path in caltech_candidates)
            + ". Place the extracted Caltech dataset under "
            + f"{project_root / 'data' / 'source_datasets' / '256_ObjectCategories'}."
        )
    multilingual_dir, multilingual_candidates = _preferred_dataset_dir(
        project_root,
        "multilingual-image-text-translation",
        "multilingual-image-text-translation",
    )
    if multilingual_dir is None:
        raise FileNotFoundError(
            "Missing multilingual image-text source directory. Checked: "
            + ", ".join(str(path) for path in multilingual_candidates)
            + ". Place the downloaded dataset under "
            + f"{project_root / 'data' / 'source_datasets' / 'multilingual-image-text-translation'}."
        )
    clock_dir, clock_candidates = _preferred_dataset_dir(
        project_root,
        "time-image-datasetclassification",
        "archive (1)",
    )
    if clock_dir is None:
        raise FileNotFoundError(
            "Missing clock-time image source directory. Checked: "
            + ", ".join(str(path) for path in clock_candidates)
            + ". Place the extracted dataset under "
            + f"{project_root / 'data' / 'source_datasets' / 'time-image-datasetclassification'}."
        )

    metadata: dict[str, Any] = {
        "sources": {
            "cifar10": str(cifar_dir),
            "caltech256": str(caltech_dir),
            "multilingual_image_text": str(multilingual_dir),
            "clock_time_reading": str(clock_dir),
        },
        "staging_parameters": {
            "image_tier": image_tier,
            "cifar_samples_per_class": samples_per_class,
            "caltech_samples_per_class": samples_per_class,
            "multilingual_image_text_samples_per_language": samples_per_class,
            "clock_time_samples_per_label": clock_samples_per_label,
            "cifar_total_images": len(CIFAR10_LABELS) * samples_per_class,
            "caltech_total_images": len(CALTECH256_SELECTED_CLASSES) * samples_per_class,
            "multilingual_image_text_total_images": len(MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES) * samples_per_class,
            "clock_time_total_images": len(CLOCK_TIME_LABELS) * clock_samples_per_label,
            "clock_time_source_split": "test",
        },
        "cifar10": [],
        "caltech256": [],
        "multilingual_image_text": [],
        "clock_time_reading": [],
    }
    scenarios: list[dict[str, Any]] = []

    meta = _load_cifar_batch(cifar_dir / "batches.meta")
    label_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in meta[b"label_names"]]
    test_batch = _load_cifar_batch(cifar_dir / "test_batch")
    test_data = test_batch[b"data"]
    test_labels = test_batch[b"labels"]
    selected_by_label: dict[int, list[int]] = {}
    for index, label_idx in enumerate(test_labels):
        selected_by_label.setdefault(label_idx, [])
        if len(selected_by_label[label_idx]) < samples_per_class:
            selected_by_label[label_idx].append(index)
        if (
            len(selected_by_label) == len(label_names)
            and all(len(indices) >= samples_per_class for indices in selected_by_label.values())
        ):
            break

    for label_idx, label_name in enumerate(label_names):
        image_indices = selected_by_label.get(label_idx, [])
        if len(image_indices) < samples_per_class:
            raise ValueError(
                f"Unable to collect {samples_per_class} CIFAR samples for label {label_name}; found only {len(image_indices)}."
            )
        for sample_rank, image_index in enumerate(image_indices[:samples_per_class], start=1):
            output_path = cifar_root / label_name / f"{label_name}_sample_{image_index:05d}.png"
            _render_cifar_png(test_data[image_index].tolist(), output_path)
            rel_path = output_path.relative_to(project_root).as_posix()
            sample_meta = {
                "dataset": "cifar10",
                "label": label_name,
                "label_index": label_idx,
                "source_split": "test_batch",
                "source_index": image_index,
                "sample_rank_within_label": sample_rank,
                "relative_image_path": rel_path,
                "candidate_labels": CIFAR10_LABELS,
            }
            metadata["cifar10"].append(sample_meta)
            _save_json(output_path.with_suffix(".json"), sample_meta)
            scenarios.append(
                _image_classification_scenario(
                    scenario_id=f"image_classification_cifar10__{label_name}__sample_{image_index:05d}",
                    title=f"CIFAR-10 Closed-Set Classification {label_name.title()} Sample {sample_rank}",
                    use_case_id="image_classification_cifar10",
                    use_case_title="CIFAR-10 Closed-Set Image Classification",
                    family="image_classification_cifar10",
                    image_rel_path=rel_path,
                    correct_label=label_name,
                    candidate_labels=CIFAR10_LABELS,
                    dataset_name="cifar10",
                    source_reference=f"test_batch index {image_index}",
                )
            )

    caltech_candidates = sorted(CALTECH256_SELECTED_CLASSES.items())
    for folder_name, canonical_label in caltech_candidates:
        source_dir = caltech_dir / folder_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing expected Caltech 256 class directory: {source_dir}")
        source_images = sorted(
            [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if len(source_images) < samples_per_class:
            raise ValueError(
                f"Unable to collect {samples_per_class} Caltech samples for label {canonical_label}; found only {len(source_images)}. "
                f"The requested image tier is {image_tier!r}."
            )
        for sample_rank, source_image in enumerate(source_images[:samples_per_class], start=1):
            output_path = caltech_root / canonical_label.replace(" ", "_") / source_image.name
            _render_caltech_jpeg(source_image, output_path)
            rel_path = output_path.relative_to(project_root).as_posix()
            sample_meta = {
                "dataset": "caltech256",
                "label": canonical_label,
                "source_category": folder_name,
                "source_file": source_image.name,
                "sample_rank_within_label": sample_rank,
                "relative_image_path": rel_path,
                "candidate_labels": list(CALTECH256_SELECTED_CLASSES.values()),
            }
            metadata["caltech256"].append(sample_meta)
            _save_json(output_path.with_suffix(".json"), sample_meta)
            scenarios.append(
                _image_classification_scenario(
                    scenario_id=f"image_classification_caltech256__{canonical_label.replace(' ', '_')}__sample_{sample_rank:02d}",
                    title=f"Caltech 256 Closed-Set Classification {canonical_label.title()} Sample {sample_rank}",
                    use_case_id="image_classification_caltech256",
                    use_case_title="Caltech 256 Closed-Set Image Classification",
                    family="image_classification_caltech256",
                    image_rel_path=rel_path,
                    correct_label=canonical_label,
                    candidate_labels=list(CALTECH256_SELECTED_CLASSES.values()),
                    dataset_name="caltech256",
                    source_reference=f"{folder_name}/{source_image.name}",
                )
            )

    records_path = multilingual_dir / "records.jsonl"
    images_dir = multilingual_dir / "images"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing multilingual dataset records file: {records_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing multilingual dataset images directory: {images_dir}")
    multilingual_records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records_by_language: dict[str, list[dict[str, Any]]] = {}
    for record in multilingual_records:
        records_by_language.setdefault(record["language"], []).append(record)
    available_language_counts = {
        language_code: len(records_by_language.get(language_code, []))
        for language_code in MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES
    }
    metadata["staging_parameters"]["multilingual_image_text_available_language_counts"] = available_language_counts
    metadata["staging_parameters"]["multilingual_image_text_staged_language_counts"] = {
        language_code: min(samples_per_class, available_language_counts.get(language_code, 0))
        for language_code in MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES
    }
    for language_code in MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES:
        language_records = sorted(records_by_language.get(language_code, []), key=lambda item: item["id"])
        if len(language_records) < samples_per_class:
            raise ValueError(
                f"Unable to collect {samples_per_class} multilingual image-text samples for language {language_code}; "
                f"found only {len(language_records)}. The requested image tier is {image_tier!r}."
            )
        for sample_rank, record in enumerate(language_records[:samples_per_class], start=1):
            source_image = images_dir / record["image_filename"]
            if not source_image.exists():
                raise FileNotFoundError(f"Missing multilingual source image: {source_image}")
            output_path = multilingual_root / language_code / record["image_filename"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, output_path)
            rel_path = output_path.relative_to(project_root).as_posix()
            sample_meta = {
                "dataset": "multilingual-image-text-translation",
                "id": record["id"],
                "text": record["text"],
                "language": record["language"],
                "language_name": record["language_name"],
                "iso3_code": record["iso3_code"],
                "script": record["script"],
                "flores_code": record["flores_code"],
                "sample_rank_within_language": sample_rank,
                "relative_image_path": rel_path,
            }
            metadata["multilingual_image_text"].append(sample_meta)
            _save_json(output_path.with_suffix(".json"), sample_meta)
            scenarios.append(
                _multilingual_image_text_extraction_scenario(
                    scenario_id=f"image_text_extraction_multilingual__{language_code}__sample_{sample_rank:03d}",
                    title=f"Multilingual Image Text Extraction {record['language_name']} Sample {sample_rank}",
                    image_rel_path=rel_path,
                    sample_id=record["id"],
                    reference_text=record["text"],
                    language_code=record["language"],
                    language_name=record["language_name"],
                    script=record["script"],
                )
            )

    clocks_csv_path = clock_dir / "clocks.csv"
    if not clocks_csv_path.exists():
        raise FileNotFoundError(f"Missing clock dataset CSV: {clocks_csv_path}")
    with clocks_csv_path.open("r", encoding="utf-8", newline="") as handle:
        clock_rows = list(csv.DictReader(handle))
    rows_by_label: dict[str, list[dict[str, str]]] = {}
    for row in clock_rows:
        if row["data set"] != "test":
            continue
        rows_by_label.setdefault(row["labels"], []).append(row)
    available_clock_counts = {label: len(rows_by_label.get(label, [])) for label in CLOCK_TIME_LABELS}
    metadata["staging_parameters"]["clock_time_available_label_counts"] = available_clock_counts
    metadata["staging_parameters"]["clock_time_staged_label_counts"] = {
        label: min(clock_samples_per_label, available_clock_counts.get(label, 0)) for label in CLOCK_TIME_LABELS
    }
    for label in CLOCK_TIME_LABELS:
        label_rows = sorted(rows_by_label.get(label, []), key=lambda item: item["filepaths"])
        if len(label_rows) < clock_samples_per_label:
            raise ValueError(
                f"Unable to collect {clock_samples_per_label} clock samples for label {label}; "
                f"found only {len(label_rows)} in the test split. The requested image tier is {image_tier!r}."
            )
        for sample_rank, row in enumerate(label_rows[:clock_samples_per_label], start=1):
            source_image = clock_dir / row["filepaths"]
            if not source_image.exists():
                raise FileNotFoundError(f"Missing clock source image: {source_image}")
            output_path = clock_root / label / f"{label}_sample_{sample_rank:02d}{source_image.suffix.lower()}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, output_path)
            rel_path = output_path.relative_to(project_root).as_posix()
            sample_meta = {
                "dataset": "time-image-datasetclassification",
                "label": label,
                "display_time": _time_label_to_display(label),
                "class_index": int(row["class index"]),
                "source_split": row["data set"],
                "source_file": row["filepaths"],
                "sample_rank_within_label": sample_rank,
                "relative_image_path": rel_path,
                "candidate_labels": CLOCK_TIME_LABELS,
            }
            metadata["clock_time_reading"].append(sample_meta)
            _save_json(output_path.with_suffix(".json"), sample_meta)
            scenarios.append(
                _clock_time_reading_scenario(
                    scenario_id=f"image_clock_time_reading__{label}__sample_{sample_rank:02d}",
                    title=f"Analog Clock Time Reading {label} Sample {sample_rank}",
                    image_rel_path=rel_path,
                    correct_label=label,
                    source_reference=row["filepaths"],
                )
            )

    _save_json(image_root / "dataset_manifest.json", metadata)
    return {
        "scenarios": scenarios,
        "use_case_docs": {
            "image_classification_cifar10": _write_use_case_doc(
                "CIFAR-10 Closed-Set Image Classification",
                f"Balanced {len(CIFAR10_LABELS) * samples_per_class}-image closed-set classification benchmark built from the CIFAR-10 test split.",
                "Measures how Gemma 4 performs on small, low-resolution visual inputs where edge devices may need lightweight visual triage or object recognition.",
            ),
            "image_classification_caltech256": _write_use_case_doc(
                "Caltech 256 Closed-Set Image Classification",
                f"Balanced {len(CALTECH256_SELECTED_CLASSES) * samples_per_class}-image closed-set classification benchmark built from a staged subset of Caltech 256 object categories.",
                "Measures larger-object visual recognition quality on more natural images while keeping the label space explicit and reviewable.",
            ),
            "image_text_extraction_multilingual": _write_multilingual_ocr_use_case_doc(
                summary=(
                    f"Balanced {len(MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES) * samples_per_class}-image multilingual "
                    "text-in-image extraction benchmark built from a staged subset of the "
                    "`multilingual-image-text-translation` dataset."
                ),
                rationale=(
                    "Measures how well Gemma 4 reads visible text across scripts and languages on-device, "
                    "which matters for signage, field documents, multilingual labels, and mixed-language edge workflows."
                ),
                language_codes=MULTILINGUAL_IMAGE_TEXT_LANGUAGE_CODES,
                available_counts=available_language_counts,
                staged_per_language=samples_per_class,
            ),
            "image_clock_time_reading": _write_clock_time_use_case_doc(
                summary=(
                    f"Held-out analog clock reading benchmark built from the `test` split of the "
                    f"`time-image-datasetclassification` dataset, with {clock_samples_per_label} samples staged "
                    f"for each of the {len(CLOCK_TIME_LABELS)} possible time labels."
                ),
                rationale=(
                    "Measures whether Gemma 4 can read analog clocks on-device without internet access, which is useful "
                    "for industrial gauges, control panels, legacy instrumentation, and visual state extraction on edge systems."
                ),
                samples_per_label=clock_samples_per_label,
                total_images=len(CLOCK_TIME_LABELS) * clock_samples_per_label,
            ),
        },
    }
