#!/usr/bin/env python
"""Export suite2p .npy outputs to CSVs + figures you can open without the GUI.

suite2p writes its results as .npy files (F, Fneu, spks, stat, iscell, ops) in a
`suite2p/plane0` folder. The GUI reads those — and your manual curation is saved
back into `iscell.npy`. This script reads them and produces:

  - traces_F.csv      raw fluorescence (time x cell)
  - traces_dff.csv    ΔF/F, neuropil-corrected (time x cell)
  - traces_spks.csv   deconvolved activity (time x cell)
  - cells.csv         one row per ROI: is_cell, prob, position, size, ...
  - rois.png          mean image with the accepted cells overlaid
  - traces.png        stacked ΔF/F traces

By default it exports only ROIs you accepted as cells (iscell == 1); use --all
for every ROI. Needs only numpy + matplotlib.

Usage
-----
  python export_suite2p.py PATH [--out DIR] [--all] [--neuropil 0.7]
                                [--fps 30] [--max-traces 60] [--no-open]

PATH may be:
  - a suite2p output folder (containing F.npy/stat.npy), or a parent that has
    suite2p/plane0 inside it, OR
  - a single .npy file (quick inspect: prints shape/dtype, dumps a CSV if numeric).

Example (using the GUI's venv on this Mac):
  /Users/leonor/Dev/embrio-T4CT/suite2p_venv/bin/python \
      scripts/export_suite2p.py ~/Downloads/suite2p/plane0
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_plane(path: Path) -> Path:
    """Locate the folder holding the suite2p .npy outputs."""
    for c in (path, path / "plane0", path / "suite2p" / "plane0"):
        if (c / "stat.npy").exists() and (c / "F.npy").exists():
            return c
    sys.exit(f"error: no suite2p outputs (F.npy + stat.npy) found under {path}")


def load_plane(plane: Path) -> dict:
    def L(name):
        return np.load(plane / name, allow_pickle=True)

    return dict(
        F=L("F.npy"), Fneu=L("Fneu.npy"), spks=L("spks.npy"),
        stat=L("stat.npy"), iscell=L("iscell.npy"), ops=L("ops.npy").item(),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def dff(F: np.ndarray, percentile: float = 10.0) -> np.ndarray:
    F0 = np.percentile(F, percentile, axis=1, keepdims=True)
    return (F - F0) / (np.abs(F0) + 1e-8)


def norm_img(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def save_traces_csv(path: Path, traces: np.ndarray, fps: float, ids):
    """time_s in column 0, one column per cell."""
    T = traces.shape[1]
    t = (np.arange(T) / fps)[:, None]
    data = np.hstack([t, traces.T])
    header = "time_s," + ",".join(f"cell{i}" for i in ids)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6g")


def save_cells_csv(path: Path, stat, iscell, ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "is_cell", "prob", "y", "x", "npix", "radius", "aspect_ratio"])
        for i in ids:
            s = stat[i]
            med = s.get("med", [np.nan, np.nan])
            w.writerow([i, int(iscell[i, 0]), round(float(iscell[i, 1]), 4),
                        int(med[0]), int(med[1]),
                        int(s.get("npix", len(s["ypix"]))),
                        round(float(s.get("radius", np.nan)), 2),
                        round(float(s.get("aspect_ratio", np.nan)), 2)])


def plot_rois(path: Path, ops: dict, stat, ids):
    import matplotlib.pyplot as plt

    Ly, Lx = ops["Ly"], ops["Lx"]
    bg = ops.get("meanImgE", ops.get("meanImg"))
    bg = norm_img(bg) if bg is not None else np.zeros((Ly, Lx))
    rng = np.random.default_rng(0)
    overlay = np.zeros((Ly, Lx, 4), np.float32)
    for i in ids:
        s = stat[i]
        overlay[s["ypix"], s["xpix"], :3] = rng.random(3)
        overlay[s["ypix"], s["xpix"], 3] = 0.55
    plt.figure(figsize=(8, 8))
    plt.imshow(bg, cmap="gray")
    plt.imshow(overlay)
    plt.title(f"suite2p ROIs (n={len(ids)})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_traces(path: Path, traces: np.ndarray, fps: float, n: int):
    import matplotlib.pyplot as plt

    tr = traces[:n]
    t = np.arange(tr.shape[1]) / fps
    spacing = float(np.nanpercentile(tr, 99)) * 1.2 or 1.0
    plt.figure(figsize=(11, max(4, 0.18 * len(tr))))
    for i, row in enumerate(tr):
        plt.plot(t, row + i * spacing, color="k", lw=0.5)
    plt.xlabel("time (s)")
    plt.yticks([])
    plt.title(f"ΔF/F traces (n={len(tr)})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Single-file quick inspect
# --------------------------------------------------------------------------- #
def inspect_npy(path: Path):
    arr = np.load(path, allow_pickle=True)
    print(f"{path.name}: dtype={arr.dtype}, shape={arr.shape}")
    if arr.dtype == object or arr.ndim == 0:
        item = arr.item() if arr.ndim == 0 else arr
        if isinstance(item, dict):
            print("  dict keys:", ", ".join(map(str, item.keys())))
        else:
            print(f"  object array of {len(arr)} items "
                  f"(e.g. suite2p 'stat' — list of per-ROI dicts)")
            if len(arr) and isinstance(arr[0], dict):
                print("  item[0] keys:", ", ".join(map(str, arr[0].keys())))
        return
    if arr.ndim <= 2:
        out = path.with_suffix(".csv")
        np.savetxt(out, np.atleast_2d(arr), delimiter=",", fmt="%.6g")
        print(f"  -> wrote {out}")
    else:
        print("  (>2D array; not dumping to CSV)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Export suite2p .npy outputs to CSV + figures.")
    ap.add_argument("path", help="suite2p output folder, or a single .npy file")
    ap.add_argument("--out", help="output directory (default: <plane>/export)")
    ap.add_argument("--all", action="store_true", help="include all ROIs, not just accepted cells")
    ap.add_argument("--neuropil", type=float, default=0.7, help="neuropil coefficient for ΔF/F (default 0.7)")
    ap.add_argument("--fps", type=float, default=None, help="frame rate (default: from ops['fs'])")
    ap.add_argument("--max-traces", type=int, default=60, help="cells to draw in traces.png (default 60)")
    ap.add_argument("--no-open", action="store_true", help="don't open the output folder when done")
    args = ap.parse_args()

    path = Path(args.path).expanduser()
    if not path.exists():
        sys.exit(f"error: path not found: {path}")

    if path.is_file() and path.suffix == ".npy":
        inspect_npy(path)
        return

    plane = find_plane(path)
    d = load_plane(plane)
    out = Path(args.out).expanduser() if args.out else plane / "export"
    out.mkdir(parents=True, exist_ok=True)
    fps = args.fps or float(d["ops"].get("fs", 30.0))

    iscell = d["iscell"]
    ids = list(range(len(d["stat"]))) if args.all else \
        [i for i in range(len(d["stat"])) if iscell[i, 0] > 0]
    if not ids:
        sys.exit("error: no cells selected (try --all to export every ROI)")

    F = d["F"][ids]
    Fcorr = F - args.neuropil * d["Fneu"][ids]
    traces_dff = dff(Fcorr)

    save_traces_csv(out / "traces_F.csv", F, fps, ids)
    save_traces_csv(out / "traces_dff.csv", traces_dff, fps, ids)
    save_traces_csv(out / "traces_spks.csv", d["spks"][ids], fps, ids)
    save_cells_csv(out / "cells.csv", d["stat"], iscell, ids)
    plot_rois(out / "rois.png", d["ops"], d["stat"], ids)
    plot_traces(out / "traces.png", traces_dff, fps, args.max_traces)

    n_total = len(d["stat"])
    print(f"suite2p plane: {plane}")
    print(f"ROIs: {len(ids)} exported "
          f"({'all' if args.all else f'accepted cells; {n_total} total ROIs'}), "
          f"{F.shape[1]} frames @ {fps:g} fps")
    print(f"wrote -> {out}")
    for f in ("traces_F.csv", "traces_dff.csv", "traces_spks.csv", "cells.csv",
              "rois.png", "traces.png"):
        print("   ", f)

    if not args.no_open and sys.platform == "darwin":
        try:
            subprocess.run(["open", str(out)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
