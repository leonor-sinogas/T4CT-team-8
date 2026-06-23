# Local suite2p GUI & exporting results

The heavy analysis runs on **Colab** (see the [main README](../README.md)). The
suite2p **GUI** is a desktop PyQt app — it needs a display, so it **cannot run in
Colab**. Use it locally to **inspect and curate** the ROIs/traces, then export.

Typical flow:

```
Colab: run pipeline  ->  download suite2p/plane0 folder  ->  local GUI: curate  ->  export to CSV/PNG
```

---

## 1. Install & launch the GUI

suite2p needs **Python 3.9–3.12** (not 3.13 — `numba` doesn't support it yet) in
its own environment.

### Easiest — the launcher script

```
scripts/launch_suite2p_gui.command
```

Double-click it in Finder (or run it in Terminal). On first run it creates a venv
at `~/suite2p_venv`, installs `suite2p[gui]` (a few minutes), and opens the GUI;
after that it just launches. To reuse a different venv:

```
SUITE2P_VENV=/path/to/venv scripts/launch_suite2p_gui.command
```

### Manual

```bash
brew install python@3.12                 # if you don't already have 3.12
python3.12 -m venv ~/suite2p_venv
~/suite2p_venv/bin/pip install -U pip
~/suite2p_venv/bin/pip install "suite2p[gui]"
~/suite2p_venv/bin/python -m suite2p     # opens the GUI
```

The `pynwb not installed` line at launch is harmless (it only disables NWB export).

> **On this project's Mac** the GUI venv already exists at
> `/Users/leonor/Dev/embrio-T4CT/suite2p_venv`, so you can launch directly with
> `/Users/leonor/Dev/embrio-T4CT/suite2p_venv/bin/python -m suite2p`.

---

## 2. Load results in the GUI

`File → Load processed data` → select the `plane0` folder (the one containing
`stat.npy`, `F.npy`, …). Toggle cells accept/reject to curate — your choices are
saved back into `iscell.npy`, and the exporter respects them.

---

## 3. Export to openable files

suite2p stores everything as `.npy`. Convert to CSV (Excel/Numbers) + figures
(Preview):

```bash
<venv>/bin/python scripts/export_suite2p.py "PATH/TO/plane0"
```

Outputs are written to `PATH/TO/plane0/export/` (Finder opens it automatically on
macOS):

| File | Contents |
|---|---|
| `rois.png` | Mean image with accepted cells overlaid |
| `traces.png` | Stacked ΔF/F traces |
| `cells.csv` | One row per cell: `is_cell, prob, y, x, npix, radius, aspect_ratio` |
| `traces_F.csv` | Raw fluorescence — `time_s` + one column per cell |
| `traces_dff.csv` | ΔF/F, neuropil-corrected |
| `traces_spks.csv` | Deconvolved activity (spikes) |

By default only ROIs you **accepted as cells** in the GUI are exported.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--all` | off | Export every ROI, not just accepted cells |
| `--neuropil COEF` | `0.7` | Neuropil subtraction for ΔF/F (`0` = none) |
| `--fps N` | from `ops['fs']` | Frame rate for the time axis |
| `--max-traces N` | `60` | Cells drawn in `traces.png` (CSVs always include all) |
| `--out DIR` | `<plane0>/export` | Output directory |
| `--no-open` | off | Don't open the folder in Finder when done |

### Quick-inspect a single `.npy`

```bash
<venv>/bin/python scripts/export_suite2p.py PATH/TO/F.npy
```

Prints shape/dtype and dumps a CSV for numeric arrays, or lists the keys for
structured files like `stat.npy` / `ops.npy`.

---

## Notes

- **Example run (this project):** 465 accepted cells out of 2647 ROIs, 2000
  frames @ 30 fps → `traces_dff.csv` is 466 columns × 2000 rows (~7.5 MB).
- Trace CSVs can get large (hundreds of cells × thousands of frames); they open
  in Numbers/Excel but may be slow. Still well within Excel's 16,384-column limit.
- **Don't commit** the `plane0` data or the `export/` outputs — they're large and
  data-specific. The repo `.gitignore` already excludes `*.npy`, `data/`, etc.
