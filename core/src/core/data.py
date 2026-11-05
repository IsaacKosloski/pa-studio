import math
from pathlib import Path

import numpy as np


def load_iq_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PA input x and output y as complex arrays.
    Expects a CSV with a header and columns Xreal, Ximg, Yreal, Yimg.
    Expects many rows.
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got shape {data.shape}")

    x = data[:, 0] + 1j * data[:, 1]
    y = data[:, 2] + 1j * data[:, 3]

    return x, y


def split_contiguous(
    x: np.ndarray,
    y: np.ndarray,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Split x and y into contiguous train, validation and test blocks."""
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    if np.shape(x) != np.shape(y):
        raise ValueError("x and y must have the same shape")

    n = len(x)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)

    x_train = x[:n_train]
    x_val = x[n_train : n_train + n_val]
    x_test = x[n_train + n_val :]

    y_train = y[:n_train]
    y_val = y[n_train : n_train + n_val]
    y_test = y[n_train + n_val :]

    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))


SPLITS = ("train", "val", "test")


def _load_iq_columns(path: Path) -> np.ndarray:
    """Load a two-column CSV (header + I, Q) as a complex array."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected 2 columns (I, Q) in {path}, got shape {data.shape}")
    iq: np.ndarray = data[:, 0] + 1j * data[:, 1]
    return iq


def load_opendpd_split(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split of an OpenDPD-format dataset.

    The OpenDPD format stores each split in two files,
    ``{split}_input.csv`` and ``{split}_output.csv``, each with a header and
    two columns (I, Q).

    Args:
        directory: Dataset folder (e.g. characterization/data/apa_200mhz).
        split: One of "train", "val", "test".

    Returns:
        (x, y): PA input and output as complex arrays.

    Raises:
        ValueError: If split is unknown, a file is malformed, or input and
            output have different lengths.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    x = _load_iq_columns(directory / f"{split}_input.csv")
    y = _load_iq_columns(directory / f"{split}_output.csv")
    if x.shape != y.shape:
        raise ValueError(
            f"Input and output of split {split!r} differ in length: "
            f"{x.shape} and {y.shape}"
        )
    return x, y


def load_opendpd_dataset(
    directory: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the train, validation and test splits of an OpenDPD-format dataset.

    Args:
        directory: Dataset folder.

    Returns:
        A dict mapping "train", "val" and "test" to (x, y) pairs, in the
        provider's contiguous order.
    """
    return {split: load_opendpd_split(directory, split) for split in SPLITS}
