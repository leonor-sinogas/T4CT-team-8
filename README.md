# 🧠 T4CT — Two-Photon Ca²⁺ Imaging Pipeline (Team 8)

Hackathon project. **Input:** a two-photon imaging time series of neural Ca²⁺
activity in mouse primary motor cortex (10,000 frames @ 30 fps, ~5 GB TIFF).
**Goal:** denoise → motion-correct → **segment neuron spatial footprints** →
**extract their Ca²⁺ activity traces**. The deliverable is a **Google Colab**
demo running on a free Tesla **T4** GPU.

> Dataset from Khan, Dutta, Scott et al., *Site-specific seeding of Lewy
> pathology…*, **Nat Commun 15, 10775 (2024)**.

## ▶️ Run it now

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonor-sinogas/T4CT-team-8/blob/main/notebooks/T4CT_demo.ipynb)

Click the badge → `Runtime → Change runtime type → T4 GPU` → `Runtime → Run all`.
On the **first** run the install cell restarts the runtime once (suite2p needs a
fresh numpy/numba) — just **Run all again** when it returns. The notebook then
runs the **whole pipeline on a built-in synthetic movie**, so it works *today* —
before the real recording is shared. Swap in the real TIFF on workshop day
(the *Load the data* cell).

## 🎯 Milestones (from the brief)

- **Minimum** — segment spatial footprints of cells + extract their Ca²⁺ dynamics.
- **Intermediate** — denoise + motion-correct the video for better segmentation.
- **Advanced** — improve deep-learning denoising using priors (Ca²⁺ dynamics
  statistics, imaging PSF).

## 🗂️ Layout

```
T4CT-team-8/
├── notebooks/
│   └── T4CT_demo.ipynb     # ← the Colab demo (entry point)
├── src/t4ct/               # importable package (edit here, keep the notebook thin)
│   ├── data.py             # TIFF IO, summary images, synthetic movie generator
│   ├── motion.py           # rigid motion-correction baseline (suite2p for full data)
│   ├── denoise.py          # temporal/PCA baselines + DeepCAD hook (advanced)
│   ├── segment.py          # suite2p pipeline + correlation-image baseline
│   ├── traces.py           # trace extraction, ΔF/F, neuropil correction
│   └── viz.py              # summary views, footprint overlays, stacked traces
├── data/                   # the recording (gitignored — use Drive, see below)
├── models/                 # checkpoints (gitignored)
└── requirements.txt        # suite2p + tifffile (torch etc. are preinstalled on Colab)
```

## 🧰 Approach & references

The reference pipelines do motion correction + source extraction + deconvolution
out of the box. We default to **suite2p** (easiest to `pip install` and run
headless on Colab — only its GUI needs a display, which we don't use); **CaImAn**
is a strong alternative but harder to install. suite2p is officially supported on
Colab — see MouseLand's [run_suite2p_colab](https://colab.research.google.com/github/MouseLand/suite2p/blob/main/jupyter/run_suite2p_colab_2021.ipynb).

- **suite2p** — https://github.com/MouseLand/suite2p (our engine: `segment.run_suite2p`)
- **CaImAn** — https://github.com/flatironinstitute/caiman
- **Motion correction** — NoRMCorre, Pnevmatikakis & Giovannucci, *J. Neurosci. Methods* (2017)
- **Denoising (advanced)** — DeepInterpolation (Lecoq et al., *Nat. Methods* 2021,
  s41592-021-01225-0) and DeepCAD-RT (Li et al., *Nat. Methods* 2021, s41592-021-01285-2)

## 📦 Data

The raw TIFF (~5 GB) is shared via Box on workshop day. **Don't commit it** —
put it in Google Drive and mount it in the notebook (cell 4). For the 5 GB file
use `data.load_tiff(path, memmap=True)` to avoid blowing up Colab RAM, or let
suite2p read the TIFF directly via `segment.run_suite2p(tiff_path, ...)`.

## 📤 Viewing / exporting suite2p results

suite2p saves its outputs as `.npy` files in a `suite2p/plane0` folder, opened
and curated in the suite2p GUI. To turn them into things you can open **without**
the GUI — CSVs (Excel/Numbers) and figures (Preview) — run:

```
python scripts/export_suite2p.py PATH        # PATH = the plane0 folder (or its parent)
```

It writes `traces_F.csv`, `traces_dff.csv`, `traces_spks.csv`, `cells.csv`,
`rois.png` and `traces.png` into `<plane0>/export/`. By default it exports only
the ROIs you accepted as cells in the GUI (`--all` for every ROI). Needs only
numpy + matplotlib. Point it at a single `.npy` for a quick inspect/CSV dump.

## 👥 Working as a team

**Code** is shared through this GitHub repo — each person runs their own Colab
runtime and pulls the latest (the notebook's cell 2 runs `git pull` for you).
Edit code in `src/t4ct/`, commit on a branch / open a PR, keep the notebook thin.

**Real-time co-editing of the notebook** (Google-Docs style):
1. Open in Colab → `File → Save a copy in Drive`.
2. **Share** that Drive copy with teammates — you can all edit/run it live.
3. Fold changes back into the repo with `File → Save a copy in GitHub`.

**Big files / datasets:** Google Drive (cell 4), not git.

## 🔧 Local editing (optional)

No virtualenv needed — all execution happens on Colab. To edit the package in an
IDE, just open the repo (plain Python). Only install deps if you want to *run*
locally: `pip install -r requirements.txt` (note: `torch` is intentionally not
pinned — Colab ships its own CUDA build).
