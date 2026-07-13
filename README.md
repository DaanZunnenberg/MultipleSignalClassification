# MusicSSVD

Subspace-based (MUSIC / ESPRIT / Matrix Pencil) time-delay estimation for
IEEE 802.11n WiFi L-LTF signals.

## Layout

- `src/musicssvd/` — the installable package
  - `generator.py` — `DataGenerator`: L-LTF signal/channel generation and transmission
  - `processor.py` — `DataProcessor`: TDE algorithms (cross-correlation, MUSIC, ESPRIT, matrix pencil)
  - `evaluator.py` — `Evaluator`: accuracy metrics against ground truth
  - `plotter.py` — `Plotter`: result visualization
  - `music.py` — `MUSICTDE`: high-level facade wiring the above together
- `scripts/run_tde.py` — end-to-end demo entry point
- `examples/` — standalone experiment scripts (template-length scans, time-varying/Doppler channel demo)

## Install

```
pip install -e .
```

## Run

```
python scripts/run_tde.py
```
