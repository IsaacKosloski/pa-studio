import numpy as np


def estimate_delay(x: np.ndarray, y: np.ndarray) -> int:
    """Delay of y relative to x, in samples, from the cross-correlation peak."""
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape. Got {x.shape} and {y.shape}"
        )

    n = len(x)
    if n == 0:
        raise ValueError("Input signals cannot be empty.")

    r = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(x)))

    k = int(np.argmax(np.abs(r)))

    if k > n // 2:
        k -= n

    return k


def find_period(x: np.ndarray, candidates: list[int]) -> tuple[int, float]:
    """Find the candidate period P for which the signal best repeats itself."""
    if not candidates:
        raise ValueError("The candidates list cannot be empty.")

    n = len(x)
    if any(p < 1 or p >= n for p in candidates):
        raise ValueError(f"All candidates must be in [1, {n - 1}].")

    max_abs_x = np.max(np.abs(x))
    if max_abs_x == 0:
        raise ValueError("Signal has zero energy, cannot compute normalized error.")

    best_period = -1
    min_error = float("inf")

    for p in candidates:
        err = np.max(np.abs(x[:-p] - x[p:]))

        if err < min_error:
            min_error = float(err)
            best_period = p

    return best_period, min_error / float(max_abs_x)
