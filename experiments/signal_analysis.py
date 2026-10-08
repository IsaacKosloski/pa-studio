"""Signal and time-series analysis of a PA dataset (roadmap step 1.6).

Runs every function of core.analysis on a measured PA input/output pair and
writes figures plus a results.json to an output folder.

Usage (from the repository root):
    uv run --group experiments python experiments/signal_analysis.py
    uv run --group experiments python experiments/signal_analysis.py \
        --data characterization/data/dadosIniciais.csv \
        --out docs/analysis/dados-iniciais

The sampling rate of dadosIniciais.csv is unknown, so frequencies are
normalized (cycles/sample) unless --fs is given.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from core.analysis.envelope import ccdf, papr_db  # noqa: E402
from core.analysis.nonlinearity import am_am_am_pm  # noqa: E402
from core.analysis.quality import estimate_delay, find_period  # noqa: E402
from core.analysis.spectrum import occupied_bandwidth, psd  # noqa: E402
from core.analysis.timeseries import (  # noqa: E402
    acf_pacf,
    linear_gain_residual,
    ljung_box,
    stationarity,
)
from core.data import load_iq_csv  # noqa: E402

Results = dict[str, Any]

PERIOD_TOLERANCE = 0.1  # normalized repetition error accepted as "periodic"


def save(fig: Any, out: Path, name: str) -> None:
    """Save a figure as PNG and close it."""
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=120)
    plt.close(fig)


def rms_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Power of a relative to the power of b, in dB."""
    return float(10 * np.log10(np.mean(np.abs(a) ** 2) / np.mean(np.abs(b) ** 2)))


# ---------------------------------------------------------------------------
# 1. Data quality
# ---------------------------------------------------------------------------
def analyze_quality(x: np.ndarray, y: np.ndarray, res: Results) -> None:
    """Integrity, DC offset, x->y delay and periodicity."""
    n = len(x)
    res["n_samples"] = n
    res["has_nan"] = bool(np.isnan(x).any() or np.isnan(y).any())
    res["dc_x_rel_rms"] = float(abs(np.mean(x)) / np.sqrt(np.mean(np.abs(x) ** 2)))
    res["dc_y_rel_rms"] = float(abs(np.mean(y)) / np.sqrt(np.mean(np.abs(y) ** 2)))
    res["delay_samples"] = estimate_delay(x, y)

    # Every multiple of the true period is also a period, so test candidates
    # in ascending order and keep the FIRST one with a small error.
    candidates = [p for p in range(1000, n // 2 + 1) if n % p == 0]
    errors = {p: find_period(x, [p])[1] for p in candidates}
    periodic = [p for p, e in errors.items() if e < PERIOD_TOLERANCE]
    if periodic:
        period = min(periodic)
        res["period"] = period
        res["period_error"] = errors[period]
        res["n_repetitions"] = n // period
    else:
        res["period"] = None


# ---------------------------------------------------------------------------
# 2. Envelope statistics
# ---------------------------------------------------------------------------
def analyze_envelope(x: np.ndarray, y: np.ndarray, res: Results, out: Path) -> None:
    """PAPR and CCDF of input and output."""
    res["papr_x_db"] = papr_db(x)
    res["papr_y_db"] = papr_db(y)
    thresholds = np.linspace(0, 12, 121)
    c_x = ccdf(x, thresholds)
    c_y = ccdf(y, thresholds)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(thresholds, np.maximum(c_x, 1e-6), label="input x")
    ax.semilogy(thresholds, np.maximum(c_y, 1e-6), "--", label="output y")
    ax.set_xlabel("Power above mean (dB)")
    ax.set_ylabel("CCDF")
    ax.set_ylim(1e-5, 1.1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save(fig, out, "ccdf")


# ---------------------------------------------------------------------------
# 3. Spectrum
# ---------------------------------------------------------------------------
def analyze_spectrum(
    x: np.ndarray, y: np.ndarray, fs: float | None, res: Results, out: Path
) -> None:
    """Welch PSD of input and output, and occupied bandwidth."""
    f, p_x = psd(x, fs=fs)
    _, p_y = psd(y, fs=fs)

    # At 99 % a strong narrowband signal hides the distortion; at 99.9 % the
    # spectral regrowth of the output shows up.
    for frac in (0.99, 0.999):
        for name, p in (("x", p_x), ("y", p_y)):
            lo, hi, bw = occupied_bandwidth(f, p, frac)
            res[f"obw_{frac}_{name}"] = {"low": lo, "high": hi, "bandwidth": bw}

    # Two strongest spectral lines of the input (exact FFT, full record).
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(x)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1 / fs if fs else 1.0))
    top = np.sort(np.argsort(spectrum)[-2:])
    res["main_tones"] = [float(freqs[k]) for k in top]

    # Each PSD normalized to its own peak, so shapes are comparable.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f, 10 * np.log10(p_x / p_x.max()), label="input x", lw=0.8)
    ax.plot(f, 10 * np.log10(p_y / p_y.max()), label="output y", lw=0.8)
    ax.set_xlabel("Frequency (Hz)" if fs else "Normalized frequency (cycles/sample)")
    ax.set_ylabel("PSD (dB, normalized to peak)")
    ax.set_ylim(-90, 5)
    ax.set_xlim(-0.2 * (fs or 1.0), 0.2 * (fs or 1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, out, "psd")


# ---------------------------------------------------------------------------
# 4. Static nonlinearity
# ---------------------------------------------------------------------------
def analyze_nonlinearity(x: np.ndarray, y: np.ndarray, res: Results, out: Path) -> None:
    """AM/AM, AM/PM and gain compression."""
    amp_in, am_am, am_pm = am_am_am_pm(x, y, n_bins=50)
    gain_db = 20 * np.log10(am_am / amp_in)
    res["small_signal_gain_db"] = float(gain_db[0])
    res["gain_max_db"] = float(gain_db.max())
    res["gain_min_db"] = float(gain_db.min())
    res["gain_variation_db"] = float(gain_db.max() - gain_db.min())
    res["am_pm_span_deg"] = float(am_pm.max() - am_pm.min())

    rng = np.random.default_rng(0)
    pick = rng.choice(len(x), size=min(5000, len(x)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(np.abs(x[pick]), np.abs(y[pick]), s=1, alpha=0.3, label="samples")
    axes[0].plot(amp_in, am_am, "k", lw=1.5, label="binned mean")
    axes[0].set(xlabel="|x|", ylabel="|y|", title="AM/AM")
    phase = np.degrees(np.angle(y[pick] * np.conj(x[pick])))
    axes[1].scatter(np.abs(x[pick]), phase, s=1, alpha=0.3)
    axes[1].plot(amp_in, am_pm, "k", lw=1.5)
    axes[1].set(xlabel="|x|", ylabel="phase(y) - phase(x) (deg)", title="AM/PM")
    axes[2].plot(amp_in, gain_db, "k")
    axes[2].set(xlabel="|x|", ylabel="gain (dB)", title="Gain vs input amplitude")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(markerscale=5)
    save(fig, out, "am_am_am_pm")


# ---------------------------------------------------------------------------
# 5. Time series (on the linear-gain residual)
# ---------------------------------------------------------------------------
def analyze_timeseries(x: np.ndarray, y: np.ndarray, res: Results, out: Path) -> None:
    """Stationarity, ACF/PACF and whiteness of the linear-gain residual."""
    r = linear_gain_residual(x, y)
    # Residual power relative to the output = NMSE of the gain-only model.
    res["linear_gain_nmse_db"] = rms_ratio(r, y)

    series = {
        "x_real": x.real,
        "x_imag": x.imag,
        "x_abs": np.abs(x),
        "r_real": r.real,
    }
    res["stationarity"] = {k: vars(stationarity(v)) for k, v in series.items()}

    nlags = 40
    acf_r, pacf_r = acf_pacf(r.real, nlags)
    p_values = ljung_box(r.real, [5, 10, 20]).tolist()
    res["ljung_box_r_real"] = dict(zip(["5", "10", "20"], p_values, strict=True))
    res["acf_r_real_lag1_to_5"] = acf_r[1:6].tolist()

    bound = 1.96 / np.sqrt(len(x))  # 95 % band for a white series
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, title in ((axes[0], acf_r, "ACF"), (axes[1], pacf_r, "PACF")):
        ax.stem(range(nlags + 1), values)
        ax.axhspan(-bound, bound, alpha=0.2, color="gray")
        ax.set(xlabel="lag (samples)", title=f"{title} of Re(residual)")
        ax.grid(True, alpha=0.3)
    save(fig, out, "acf_pacf_residual")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=Path("characterization/data/dadosIniciais.csv")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/analysis/dados-iniciais")
    )
    parser.add_argument("--fs", type=float, default=None, help="sampling rate (Hz)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    x, y = load_iq_csv(args.data)
    res: Results = {"dataset": str(args.data), "fs": args.fs}

    analyze_quality(x, y, res)
    analyze_envelope(x, y, res, args.out)
    analyze_spectrum(x, y, args.fs, res, args.out)
    analyze_nonlinearity(x, y, res, args.out)
    analyze_timeseries(x, y, res, args.out)

    text = json.dumps(res, indent=2, default=float)
    (args.out / "results.json").write_text(text + "\n")
    print(text)
    print(f"\nFigures and results.json written to {args.out}/")


if __name__ == "__main__":
    main()
