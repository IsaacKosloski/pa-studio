import numpy as np
import pytest
from core.analysis.spectrum import occupied_bandwidth, psd


def test_psd_complex_tone_peaks_at_its_frequency() -> None:
    n = np.arange(2**14)
    x = np.exp(2j * np.pi * 0.1 * n)
    f, pxx = psd(x, nperseg=1024)
    assert f[np.argmax(pxx)] == pytest.approx(0.1, abs=1 / 1024)


def test_occupied_bandwidth_of_a_tone_is_narrow() -> None:
    n = np.arange(2**14)
    x = np.exp(2j * np.pi * 0.1 * n)
    f, pxx = psd(x, nperseg=1024)
    f_low, f_high, bandwidth = occupied_bandwidth(f, pxx)
    assert f_low <= 0.1 <= f_high
    assert bandwidth < 0.02
