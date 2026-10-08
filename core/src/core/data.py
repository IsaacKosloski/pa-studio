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
