from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    temp_root = BACKEND_ROOT / ".pytest_tmp"
    temp_path = temp_root / uuid4().hex
    temp_path.mkdir(parents=True)

    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
