from collections.abc import Callable

import numpy as np
import pytest
from core.metrics import nmse_db, r2, rmse


@pytest.fixture
def y() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(1000) + 1j * rng.standard_normal(1000)


# Common tests for all metrics
@pytest.mark.parametrize("metric", [nmse_db, rmse, r2])
def test_metric_shape_mismatch_raises(
    metric: Callable[[np.ndarray, np.ndarray], float | np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="shape"):
        metric(np.ones(3), np.ones(1))


@pytest.mark.parametrize("metric", [nmse_db, rmse, r2])
def test_metric_zero_energy_reference_raises(
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> None:
    with pytest.raises(ValueError, match="zero energy"):
        metric(np.zeros(4), np.ones(4))


# Tests for NMSE (dB)
def test_nmse_db_zero_prediction_is_0_db(y: np.ndarray) -> None:
    assert nmse_db(y, np.zeros_like(y)) == pytest.approx(0.0)


def test_nmse_db_10_percent_gain_error_is_minus_20_db(y: np.ndarray) -> None:
    assert nmse_db(y, 1.1 * y) == pytest.approx(-20.0)


def test_nmse_db_perfect_prediction_is_minus_inf(y: np.ndarray) -> None:
    assert nmse_db(y, y) == -np.inf


def test_nmse_db_non_proportional_error_matches_hand_calculation() -> None:
    y_test = np.array([3.0, 4.0])
    y_hat = np.array([3.0, 0.0])
    assert nmse_db(y_test, y_hat) == pytest.approx(-1.938, abs=1e-3)


# Tests for RMSE
def test_rmse_perfect_prediction_is_zero(y: np.ndarray) -> None:
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_constant_real_offset(y: np.ndarray) -> None:
    offset = 2.5
    assert rmse(y, y + offset) == pytest.approx(np.abs(offset))


def test_rmse_constant_complex_offset(y: np.ndarray) -> None:
    offset = 3.0 + 4.0j
    assert rmse(y, y + offset) == pytest.approx(5.0)


def test_rmse_matches_hand_calculation() -> None:
    y_test = np.array([3.0, 4.0])
    y_hat = np.array([3.0, 0.0])
    assert rmse(y_test, y_hat) == pytest.approx(np.sqrt(8.0))


# Tests for R²
def test_r2_perfect_prediction_is_one(y: np.ndarray) -> None:
    assert r2(y, y) == pytest.approx(1.0)


def test_r2_mean_prediction_is_zero(y: np.ndarray) -> None:
    y_mean = np.mean(y)
    y_hat = np.full_like(y, fill_value=y_mean)
    assert r2(y, y_hat) == pytest.approx(0.0, abs=1e-10)


def test_r2_worse_than_mean_is_negative(y: np.ndarray) -> None:
    y_mean = np.mean(y)
    y_hat = y_mean - (y - y_mean) * 2
    assert r2(y, y_hat) < 0.0


def test_r2_zero_variance_signal() -> None:
    y_test = np.ones(10)
    assert r2(y_test, y_test) == pytest.approx(0.0)
