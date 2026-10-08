"""Time-series diagnostics: stationarity, autocorrelation and whiteness.

All functions except linear_gain_residual take a REAL-valued series. For
complex signals, analyze the real part, the imaginary part and/or the
magnitude separately.

The PA itself is best seen in the residual of a linear gain fit,
r = y - g * x (see linear_gain_residual): the raw signals mostly show the
structure of the input waveform, not of the amplifier.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


@dataclass(frozen=True)
class StationarityResult:
    """p-values of two complementary stationarity tests.

    ADF: null hypothesis = unit root (NON-stationary).
         p < 0.05 -> reject -> evidence of stationarity.
    KPSS: null hypothesis = stationary.
          p < 0.05 -> reject -> evidence of NON-stationarity.
    Stationary series: ADF p small AND KPSS p large.
    """

    adf_pvalue: float
    kpss_pvalue: float


def linear_gain_residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual of the best memoryless complex gain fit, r = y - g * x.

    g is the least-squares gain: g = <x, y> / <x, x>.

    Args:
        x: PA input (complex).
        y: PA output (complex), same shape as x.

    Returns:
        The complex residual r, same shape as x.

    Raises:
        ValueError: If the shapes differ, the inputs are empty, or x has zero
            energy.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape. Got {x.shape} and {y.shape}"
        )
    if x.size == 0:
        raise ValueError("Input signals cannot be empty.")

    energy_x = np.vdot(x, x).real
    if energy_x == 0:
        raise ValueError("Input signal x has zero energy.")

    g = np.vdot(x, y) / energy_x
    residual: np.ndarray = y - g * x
    return residual


def stationarity(z: np.ndarray) -> StationarityResult:
    """Run the ADF and KPSS stationarity tests on a real series.

    Args:
        z: Real-valued series.

    Returns:
        A StationarityResult with both p-values. KPSS p-values are clipped
        by statsmodels to its lookup table, [0.01, 0.10].

    Raises:
        ValueError: If z is empty, complex, or constant.
    """
    if z.size == 0:
        raise ValueError("Input series cannot be empty.")
    if np.iscomplexobj(z):
        raise ValueError("Stationarity tests require a real-valued series.")
    if np.all(z == z[0]):
        raise ValueError("Series has zero variance.")

    adf_p = float(adfuller(z, result_object=False)[1])

    # KPSS warns when the p-value falls outside its lookup table [0.01, 0.10]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_p = float(kpss(z, nlags="auto", result_object=False)[1])

    return StationarityResult(adf_pvalue=adf_p, kpss_pvalue=kpss_p)


def acf_pacf(z: np.ndarray, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation and partial autocorrelation of a real series.

    Args:
        z: Real-valued series.
        nlags: Number of lags (lag 0 is included, so arrays have nlags + 1
            values).

    Returns:
        (acf_values, pacf_values).

    Raises:
        ValueError: If z is empty or complex, or nlags is out of range.
    """
    if z.size == 0:
        raise ValueError("Input series cannot be empty.")
    if np.iscomplexobj(z):
        raise ValueError("ACF/PACF functions require a real-valued series.")
    if nlags < 1:
        raise ValueError(f"nlags must be at least 1, got {nlags}.")
    if nlags >= len(z):
        raise ValueError(
            f"nlags ({nlags}) must be smaller than series length ({len(z)})."
        )

    a = acf(z, nlags=nlags, fft=True)
    p = pacf(z, nlags=nlags)

    return np.asarray(a, dtype=float), np.asarray(p, dtype=float)


def ljung_box(z: np.ndarray, lags: list[int]) -> np.ndarray:
    """Ljung-Box whiteness test p-values at the given lags.

    Null hypothesis: no autocorrelation up to each lag (white series).
    p < 0.05 -> reject -> there is still temporal structure to model.

    Args:
        z: Real-valued series (typically a model residual).
        lags: Lags at which to report the test.

    Returns:
        p-values, one per lag.

    Raises:
        ValueError: If z is empty or complex, or a lag is out of range.
    """
    if z.size == 0:
        raise ValueError("Input series cannot be empty.")
    if np.iscomplexobj(z):
        raise ValueError("Ljung-Box test requires a real-valued series.")
    if not lags:
        raise ValueError("Lags list cannot be empty.")
    if any(lag < 1 or lag >= len(z) for lag in lags):
        raise ValueError("All lags must be between 1 and len(z) - 1.")

    table = acorr_ljungbox(z, lags=lags)
    p_values: np.ndarray = table["lb_pvalue"].to_numpy(dtype=float)
    return p_values
