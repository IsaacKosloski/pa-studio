import numpy as np


def am_am_am_pm(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned AM/AM and AM/PM curves of a PA."""
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape. Got {x.shape} and {y.shape}"
        )

    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    abs_x = np.abs(x)
    abs_y = np.abs(y)

    edges = np.linspace(0, np.max(abs_x), n_bins + 1)
    idx = np.digitize(abs_x, edges[1:-1])

    phase = np.degrees(np.angle(y * np.conj(x)))

    amp_in_list = []
    am_am_list = []
    am_pm_list = []

    for b in np.unique(idx):
        mask = idx == b
        amp_in_list.append(np.mean(abs_x[mask]))
        am_am_list.append(np.mean(abs_y[mask]))
        am_pm_list.append(np.mean(phase[mask]))

    return np.array(amp_in_list), np.array(am_am_list), np.array(am_pm_list)
