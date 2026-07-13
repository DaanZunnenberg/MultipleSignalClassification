# MultipleSignalClassification

Subspace-based (MUSIC / ESPRIT / Matrix Pencil) time-delay estimation for
IEEE 802.11n WiFi L-LTF signals.

## Layout

- `src/musicssvd/` — the installable package
  - `generator.py` — `DataGenerator`: L-LTF signal/channel generation and transmission
  - `processor.py` — `DataProcessor`: TDE algorithms (cross-correlation, MUSIC, ESPRIT, matrix pencil)
  - `evaluator.py` — `Evaluator`: accuracy metrics against ground truth
  - `plotter.py` — `Plotter`: result visualization
  - `music.py` — `MUSICTDE`: high-level facade wiring the above together
- `scripts/run_tde.py` — end-to-end demo entry point, runs and plots all TDE methods
- `examples/basic_usage.py` — minimal, clean example of the core API (start here)
- `examples/forward.py`, `examples/forward_scan.py`, `examples/time_varying.py` — exploratory research scripts (template-length scans, time-varying/Doppler channel demo); heavier on ad-hoc plotting, not meant as API references

## Install

```
pip install -e .
```

## Run

```
python examples/basic_usage.py
python scripts/run_tde.py
```
