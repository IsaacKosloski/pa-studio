"""Integrity and sanity checks of the APA_200MHz dataset.

The CSV files are stored with Git LFS. Where only the LFS pointers are
checked out (e.g. CI without LFS), these tests are skipped.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from core.data import load_opendpd_dataset
from core.metrics import nmse_db

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "apa_200mhz"
LFS_POINTER_PREFIX = b"version https://git-lfs"


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


requires_data = pytest.mark.skipif(
    _is_lfs_pointer(DATA_DIR / "train_input.csv"),
    reason="APA_200MHz CSVs are LFS pointers (run `git lfs pull`)",
)


@requires_data
def test_apa_200mhz_files_match_checksums() -> None:
    for line in (DATA_DIR / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        assert digest == expected, f"{name} changed"


@requires_data
def test_apa_200mhz_linear_gain_nmse_is_about_minus_19_6_db() -> None:
    data = load_opendpd_dataset(DATA_DIR)
    x_train, y_train = data["train"]
    gain = np.vdot(x_train, y_train) / np.vdot(x_train, x_train)

    assert abs(gain) == pytest.approx(1.163, abs=1e-3)
    for x, y in data.values():
        assert nmse_db(y, gain * x) == pytest.approx(-19.6, abs=0.2)
