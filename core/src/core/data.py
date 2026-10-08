from pathlib import Path

import numpy as np


def load_iq_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PA input x and output y as complex arrays.
    Expects a CSV with a header and columns Xreal, Ximg, Yreal, Yimg.
    Expects many rows.
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got shape {data.shape[1]}")
    x = data[:, 0] + 1j * data[:, 1]
    y = data[:, 2] + 1j * data[:, 3]
    return x, y
