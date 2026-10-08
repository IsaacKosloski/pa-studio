import numpy as np
import pytest
from core.analysis.envelope import ccdf, papr_db


def test_papr_db_constant_envelope_is_0_db() -> None:
    x = np.exp(1j * np.linspace(0, 20, 1000))
    assert papr_db(x) == pytest.approx(0.0, abs=1e-9)


def test_ccdf_complex_gaussian_matches_exponential_law() -> None:
    rng = np.random.default_rng(0)
    n = 10**6
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    thresholds_db = np.array([0.0, 5.0])
    expected = np.exp(-(10 ** (thresholds_db / 10)))
    np.testing.assert_allclose(ccdf(x, thresholds_db), expected, rtol=0.03)
