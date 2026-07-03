#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gemma_vllm_benchmark.generate_assets import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-tier",
        choices=["small", "medium", "large"],
        default="medium",
        help="Controls how many staged image-classification samples are generated per dataset family.",
    )
    args = parser.parse_args()
    generate(PROJECT_ROOT, image_tier=args.image_tier)
    print(f"Generated benchmark assets under {PROJECT_ROOT}")
