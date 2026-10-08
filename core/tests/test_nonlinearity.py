import numpy as np
from core.analysis.nonlinearity import am_am_am_pm


def test_am_am_am_pm_of_linear_gain() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10**5) + 1j * rng.standard_normal(10**5)
    y = 2.0 * x * np.exp(1j * 0.5)

    amp_in, am_am, am_pm_deg = am_am_am_pm(x, y, n_bins=20)

    np.testing.assert_allclose(am_am, 2.0 * amp_in, rtol=1e-9)
    np.testing.assert_allclose(am_pm_deg, np.degrees(0.5), atol=1e-6)
