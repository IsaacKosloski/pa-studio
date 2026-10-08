import numpy as np
from core.analysis.quality import estimate_delay, find_period


def test_estimate_delay_recovers_known_shift() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    assert estimate_delay(x, np.roll(x, 7)) == 7


def test_find_period_detects_repeated_block() -> None:
    rng = np.random.default_rng(0)
    block = rng.standard_normal(500)
    period, error = find_period(np.tile(block, 6), candidates=[250, 400, 500, 750])
    assert period == 500
    assert error < 1e-12
