from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow running `pytest` from a clean checkout without an editable install,
# mirroring the sys.path shim used by the entry-point scripts.
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT
