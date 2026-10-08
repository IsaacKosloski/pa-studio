import numpy as np
import pytest
from core.analysis.timeseries import (
    acf_pacf,
    linear_gain_residual,
    ljung_box,
    stationarity,
)


@pytest.fixture
def white_noise() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(5000)


def test_linear_gain_residual_is_zero_for_pure_gain() -> None:
    x = np.random.default_rng(0).standard_normal(100) + 0j
    np.testing.assert_allclose(linear_gain_residual(x, 3j * x), 0, atol=1e-12)


def test_stationarity_white_noise_is_stationary(white_noise: np.ndarray) -> None:
    result = stationarity(white_noise)
    assert result.adf_pvalue < 0.05
    assert result.kpss_pvalue > 0.05


def test_acf_of_ar1_matches_coefficient() -> None:
    rng = np.random.default_rng(0)
    e = rng.standard_normal(20000)
    z = np.zeros_like(e)
    for n in range(1, len(e)):
        z[n] = 0.8 * z[n - 1] + e[n]
    acf_values, _ = acf_pacf(z, nlags=5)
    assert acf_values[1] == pytest.approx(0.8, abs=0.02)


def test_ljung_box_does_not_reject_white_noise(white_noise: np.ndarray) -> None:
    assert np.all(ljung_box(white_noise, lags=[10, 20]) > 0.05)
