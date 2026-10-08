# APA_200MHz dataset — provenance

Measured baseband input/output of an RF power amplifier, redistributed
unmodified from the OpenDPD project.

| Item | Value |
|---|---|
| Upstream repository | https://github.com/lab-emi/OpenDPD |
| Upstream path | `datasets/APA_200MHz/` |
| Upstream commit | `256f856b2456c126cdd87d0a7ca06fe227b664fb` (2026-10-06) |
| Downloaded | 2026-10-08 |
| License | Apache License 2.0 — copy in [`LICENSE`](LICENSE) |
| Modifications | none (files copied byte for byte; see [`SHA256SUMS`](SHA256SUMS)) |

Files not redistributed: the upstream figures (`*.png`) and helper scripts
(`demod.py`, `plot_dataset.py`).

## Content

| File | Samples | Columns |
|---|---|---|
| `train_input.csv`, `train_output.csv` | 58 980 | `I,Q` |
| `val_input.csv`, `val_output.csv` | 19 662 | `I,Q` |
| `test_input.csv`, `test_output.csv` | 19 662 | `I,Q` |

The provider's split (60/20/20) is used as-is: contiguous blocks in the order
train → validation → test. Never re-split or shuffle.

## Signal and device

From [`spec.json`](spec.json): 5-carrier LTE TM3.1a (256-QAM), 20 MHz
carriers generated at 491.52 MS/s and transmitted/captured at 983.04 MS/s, so
each carrier occupies 40 MHz and the total bandwidth is 200 MHz; PAPR 10 dB.

From the upstream release notes (`docs/whats-new.md`, OpenDPDv2 entry): the
signal was "measured from a 3.5 GHz Ampleon GaN PA at 41.5 dBm average output
power". The exact part number and the amplifier topology are not stated in
the upstream repository; do not report them without a primary source.

## How to cite

Wu, Y., Singh, G. D., Beikmirza, M., de Vreede, L. C. N., Alavi, M., and
Gao, C. "OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework
for Wideband Power Amplifier Modeling and Digital Pre-Distortion." 2024 IEEE
International Symposium on Circuits and Systems (ISCAS), pp. 1–5, 2024.
doi:10.1109/ISCAS58744.2024.10558162

## Verify integrity

```bash
cd characterization/data/apa_200mhz && sha256sum -c SHA256SUMS
```
