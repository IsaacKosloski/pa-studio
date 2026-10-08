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
    """Residual of the best memoryless complex gain fit, r = y - g * x."""
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
    """Run the ADF and KPSS stationarity tests on a real series."""
    if z.size == 0:
        raise ValueError("Input series cannot be empty.")
    if np.iscomplexobj(z):
        raise ValueError("Stationarity tests require a real-valued series.")
    if np.all(z == z[0]):
        raise ValueError("Series has zero variance.")

    adf_p = float(adfuller(z)[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_p = float(kpss(z, nlags="auto")[1])

    return StationarityResult(adf_pvalue=adf_p, kpss_pvalue=kpss_p)


def acf_pacf(z: np.ndarray, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation and partial autocorrelation of a real series."""
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
    """Ljung-Box whiteness test p-values at the given lags."""
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
