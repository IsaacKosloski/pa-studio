"""Data quality checks: input/output delay and periodicity."""

import numpy as np


def estimate_delay(x: np.ndarray, y: np.ndarray) -> int:
    """Delay of y relative to x, in samples, from the cross-correlation peak.

    Uses circular cross-correlation computed with the FFT:
        r = ifft( fft(y) * conj(fft(x)) )
    The index of max|r| is the delay. Indices above N/2 represent
    negative delays (y leads x).

    Args:
        x: Reference signal (PA input).
        y: Delayed signal (PA output), same length as x.

    Returns:
        The delay in samples. Positive means y lags x; 0 means aligned.

    Raises:
        ValueError: If x and y have different shapes or are empty.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape. Got {x.shape} and {y.shape}"
        )

    n = len(x)
    if n == 0:
        raise ValueError("Input signals cannot be empty.")

    # Circular cross-correlation via FFT
    r = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(x)))

    k = int(np.argmax(np.abs(r)))

    # Wrap to a negative delay (y leads x)
    if k > n // 2:
        k -= n

    return k


def find_period(x: np.ndarray, candidates: list[int]) -> tuple[int, float]:
    """Find the candidate period P for which the signal best repeats itself.

    For each candidate P, the repetition error is
        max | x[n] - x[n + P] |
    over all valid n. A truly periodic signal has (near) zero error at its
    period. Note that every multiple of the true period is also a period:
    to find the fundamental one, test candidates in ascending order.

    Args:
        x: Signal to test.
        candidates: Periods to try, each in [1, len(x) - 1].

    Returns:
        (best_period, best_error_normalized), where the error is divided by
        max|x| so that it is scale-independent (0 = exact repetition).

    Raises:
        ValueError: If candidates is empty, a candidate is out of range, or
            the signal is identically zero.
    """
    if not candidates:
        raise ValueError("The candidates list cannot be empty.")

    n = len(x)
    if any(p < 1 or p >= n for p in candidates):
        raise ValueError(f"All candidates must be in [1, {n - 1}].")

    max_abs_x = np.max(np.abs(x))
    if max_abs_x == 0:
        raise ValueError("Signal has zero energy, cannot normalize the error.")

    best_period = -1
    min_error = float("inf")

    for p in candidates:
        # Repetition error for period p
        err = np.max(np.abs(x[:-p] - x[p:]))

        if err < min_error:
            min_error = float(err)
            best_period = p

    return best_period, min_error / float(max_abs_x)
