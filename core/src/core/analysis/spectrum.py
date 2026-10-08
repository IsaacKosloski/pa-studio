"""Spectral analysis: power spectral density and occupied bandwidth."""

import numpy as np
from scipy import signal


def psd(
    x: np.ndarray,
    fs: float | None = None,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided power spectral density estimated with Welch's method.

    A complex baseband signal has a non-symmetric spectrum, so both sides
    are returned (return_onesided=False), ordered from -fs/2 to +fs/2.

    Args:
        x: Real or complex signal.
        fs: Sampling rate in Hz. If None, frequencies are normalized
            (cycles per sample, from -0.5 to 0.5).
        nperseg: Segment length for Welch's method.

    Returns:
        (f, pxx): frequencies (ascending) and PSD values (linear, power per
        unit frequency).

    Raises:
        ValueError: If x is empty or nperseg is not positive.
    """
    if x.size == 0:
        raise ValueError("Input signal cannot be empty.")
    if nperseg <= 0:
        raise ValueError(f"nperseg must be a positive integer, got {nperseg}.")

    rate = fs if fs is not None else 1.0
    f, pxx = signal.welch(x, fs=rate, nperseg=nperseg, return_onesided=False)

    f_shifted = np.fft.fftshift(f)
    pxx_shifted = np.fft.fftshift(pxx)

    return f_shifted, pxx_shifted


def occupied_bandwidth(
    f: np.ndarray,
    pxx: np.ndarray,
    fraction: float = 0.99,
) -> tuple[float, float, float]:
    """Frequency band that contains a given fraction of the total power.

    The band is defined by cutting (1 - fraction) / 2 of the power from each
    side of the spectrum.

    Args:
        f: Frequencies in ascending order (as returned by psd).
        pxx: PSD values at f.
        fraction: Power fraction inside the band, in (0, 1).

    Returns:
        (f_low, f_high, bandwidth) with bandwidth = f_high - f_low, in the
        same unit as f.

    Raises:
        ValueError: If fraction is not in (0, 1), the shapes differ, the
            inputs are empty, or the total power is not positive.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}.")
    if f.shape != pxx.shape:
        raise ValueError(
            f"f and pxx must have the same shape. Got {f.shape} and {pxx.shape}"
        )
    if f.size == 0:
        raise ValueError("Inputs cannot be empty.")

    total_power = np.sum(pxx)
    if total_power <= 0:
        raise ValueError("PSD values must have positive total power.")

    c = np.cumsum(pxx) / total_power
    tail = (1.0 - fraction) / 2.0

    idx_low = int(np.searchsorted(c, tail))
    idx_high = int(np.searchsorted(c, 1.0 - tail))

    idx_high = min(idx_high, len(f) - 1)

    f_low = float(f[idx_low])
    f_high = float(f[idx_high])
    bandwidth = float(f_high - f_low)

    return f_low, f_high, bandwidth
