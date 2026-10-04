import numpy as np
import pytest
from characterization.memory_polynomial import MemoryPolynomial


def test_regressor_shape() -> None:
    x = np.array([1 + 1j, 2 - 1j, 0.5 + 2j, -1 + 0j])
    K = 3
    M = 2
    N = len(x)

    Phi = MemoryPolynomial(K, M)._build_regressors(x)
    # Shape test
    assert Phi.shape == (N, K * M)


def test_linear_column_equals_input() -> None:
    x = np.array([1 + 1j, 2 - 1j, 0.5 + 2j, -1 + 0j])
    K = 1
    M = 1

    Phi = MemoryPolynomial(K, M)._build_regressors(x)

    np.testing.assert_allclose(actual=Phi[:, 0], desired=x, rtol=1e-5, atol=1e-8)


def test_delayed_column_is_zero_padded() -> None:
    x = np.array([1 + 1j, 2 - 1j, 0.5 + 2j, -1 + 0j])
    K = 3
    M = 2
    x_desired = [0, x[0], x[1], x[2]]
    Phi = MemoryPolynomial(K, M)._build_regressors(x)

    np.testing.assert_allclose(
        actual=Phi[:, 1], desired=x_desired, rtol=1e-5, atol=1e-8
    )


def test_fit_predict() -> None:
    # Generates a Random signal
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100) + 1j * rng.standard_normal(100)

    K = 2
    M = 2
    model = MemoryPolynomial(K, M)

    # Creates an artificial 'y' and a well known theta
    Phi_real = model._build_regressors(x)
    theta_template = np.array([1.0, 0.5 + 0.1j, -0.2, 0.05j])
    y_real = Phi_real @ theta_template

    # Model training
    model.fit(x, y_real)
    y_predicted = model.predict(x)

    assert model.theta is not None, "The theta was not computed properly"
    np.testing.assert_allclose(
        actual=model.theta, desired=theta_template, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(actual=y_predicted, desired=y_real, rtol=1e-5, atol=1e-8)


def test_predict_raise_error() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100) + 1j * rng.standard_normal(100)

    K = 2
    M = 2
    model = MemoryPolynomial(K, M)

    with pytest.raises(AttributeError, match="must be fit"):
        model.predict(x)
