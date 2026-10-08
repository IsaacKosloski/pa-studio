"""Static nonlinearity characterization: AM/AM and AM/PM curves."""

import numpy as np


def am_am_am_pm(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned AM/AM and AM/PM curves of a PA.

    The input amplitude |x| is split into n_bins equal-width bins. In each
    bin:
      - input level: mean input amplitude |x| (not the bin center: inside a
        wide bin the samples are not centered, mostly in the low bins);
      - AM/AM: mean output amplitude |y|;
      - AM/PM: mean phase shift angle(y * conj(x)), in degrees.

    Averaging inside bins removes the spread caused by noise and memory,
    leaving the static (memoryless) behavior. The spread itself is a sign of
    memory effects and is analyzed separately.

    Args:
        x: PA input (complex).
        y: PA output (complex), same shape as x.
        n_bins: Number of amplitude bins.

    Returns:
        (amp_in, am_am, am_pm_deg): mean |x| per bin, mean |y| per bin,
        and mean phase shift per bin in degrees. Empty bins are removed.

    Raises:
        ValueError: If x and y have different shapes.
    """
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
