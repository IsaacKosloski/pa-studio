# ADR 0002: Second dataset (OpenDPD APA_200MHz) via Git LFS, with provenance

## Status
Accepted

## Context
`dadosIniciais.csv` is a two-tone excitation whose waveform repeats 18 times
(see `docs/analysis/dados-iniciais.md`): validation and test on it measure
noise generalization only, and its low PAPR does not exercise the high-peak
regime of modulated signals. The project needs a public, documented dataset
of a modulated wideband signal, so that results are reproducible and
comparable with the literature.

OpenDPD (`lab-emi/OpenDPD`, Apache-2.0) ships `datasets/APA_200MHz`: measured
baseband input/output of a 3.5 GHz GaN PA driven by a 5-carrier LTE signal
(200 MHz, 983.04 MS/s, PAPR 10 dB), already split by the provider into
contiguous train/validation/test files (58 980 / 19 662 / 19 662 samples).
The repository has no data-specific license: its root Apache-2.0 license
applies to all contents, and there is no NOTICE file.

## Decision
- Store the dataset unmodified in `characterization/data/apa_200mhz/`,
  tracked by Git LFS (`*.csv` is already an LFS pattern).
- Keep the provider's split files as-is. Never re-split or shuffle.
- Keep next to the data: `LICENSE` (verbatim Apache-2.0 copy), `spec.json`
  (verbatim), `SOURCE.md` (upstream URL, full commit hash, download date,
  citation, origin of every device detail) and `SHA256SUMS`.
- Tests verify the checksums and a sanity value (linear complex gain gives
  NMSE ≈ −19.6 dB on every split).
- CI does **not** download LFS objects. Dataset tests detect LFS pointer
  files and are skipped there; they run wherever the data is present.
- Upstream figures and helper scripts are not redistributed. Any OpenDPD code
  reused later goes to `third_party/opendpd/` with its license header.
- `dadosIniciais.csv` stays as a complementary dataset.

## Alternatives considered
- **Download at runtime from GitHub:** rejected. Upstream may change or
  disappear, breaking reproducibility of published numbers.
- **Plain git (no LFS):** rejected. Files exceed the 500 KB pre-commit limit
  and would bloat the history.
- **Re-split the concatenated data:** rejected. Using the provider's split
  keeps results comparable with OpenDPD baselines.
- **Fetch LFS objects in CI:** postponed. It consumes the repository's LFS
  bandwidth quota on every run; the skip-on-pointer tests cover local runs.

## Consequences
- Positive: reproducible, auditable data; license obligations met;
  comparability with published results; a dataset where validation measures
  real generalization.
- Negative: dataset integrity is not checked in CI (only locally); a
  contributor needs `git lfs install` and `git lfs pull`.
- Device details beyond what the upstream repository states (exact part
  number, topology) must not be reported without a primary source.
