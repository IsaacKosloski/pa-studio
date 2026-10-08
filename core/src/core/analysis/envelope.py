"""Envelope statistics of complex baseband signals: PAPR and CCDF."""

import numpy as np


def papr_db(x: np.ndarray) -> float:
    """Peak-to-average power ratio of a signal, in dB.

    PAPR = 10 * log10( max|x|^2 / mean|x|^2 ).

    Args:
        x: Real or complex signal.

    Returns:
        The PAPR in dB. A constant-envelope signal (e.g. a complex
        exponential) gives 0 dB; a real sine wave gives about 3.01 dB.

    Raises:
        ValueError: If x is empty or has zero energy.
    """
    if x.size == 0:
        raise ValueError("Input signal cannot be empty.")

    power = np.abs(x) ** 2
    average_power = np.mean(power)
    if average_power == 0:
        raise ValueError("Signal has zero energy.")

    peak_power = np.max(power)
    return float(10 * np.log10(peak_power / average_power))


def ccdf(x: np.ndarray, thresholds_db: np.ndarray) -> np.ndarray:
    """Complementary cumulative distribution of the instantaneous power.

    For each threshold t (in dB above the mean power), returns the fraction
    of samples whose instantaneous power exceeds the mean power by more
    than t:

        CCDF(t) = P( |x|^2 / mean|x|^2 > 10^(t/10) )

    Args:
        x: Real or complex signal.
        thresholds_db: Thresholds in dB relative to the mean power.

    Returns:
        Array with the same length as thresholds_db, values in [0, 1],
        non-increasing when thresholds_db is increasing.

    Raises:
        ValueError: If x has zero energy.
    """
    power = np.abs(x) ** 2
    average_power = np.mean(power)
    if average_power == 0:
        raise ValueError("Signal has zero energy.")

    normalized_power = power / average_power
    thresholds = 10 ** (thresholds_db / 10)

    return np.array([np.mean(normalized_power > t) for t in thresholds])
