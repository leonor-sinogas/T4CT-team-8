"""Neuron segmentation (spatial footprints).

Two routes:
- `run_suite2p` — the real pipeline (registration + ROI detection + deconvolution
  + traces). This covers the minimum & intermediate milestones in one call.
- `correlation_segment` — a dependency-light baseline (correlation image +
  connected components) for quick iteration before suite2p is set up.

suite2p: https://github.com/MouseLand/suite2p   CaImAn alternative: see README.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Baseline: correlation-image segmentation (no suite2p needed)
# --------------------------------------------------------------------------- #
def correlation_segment(mov: np.ndarray, min_corr: float = 0.3,
                        min_size: int = 15, max_size: int = 600
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Threshold the local correlation image and keep blobs of cell-like size.

    Returns (footprints[N, H, W], correlation_image[H, W]).
    """
    from skimage.measure import label, regionprops
    from .data import correlation_image

    ci = correlation_image(mov)
    labels = label(ci > min_corr)
    footprints = []
    for r in regionprops(labels):
        if min_size <= r.area <= max_size:
            fp = np.zeros(ci.shape, np.float32)
            ys, xs = r.coords[:, 0], r.coords[:, 1]
            fp[ys, xs] = ci[ys, xs]
            footprints.append(fp)
    return np.asarray(footprints, np.float32), ci


# --------------------------------------------------------------------------- #
# Full pipeline: suite2p
# --------------------------------------------------------------------------- #
def run_suite2p(data, save_path: str, fs: float = 30.0, tau: float = 1.0,
                diameter=None, **settings_kw) -> dict:
    """Run the suite2p pipeline (suite2p >= 1.0 API) and return its outputs.

    `data` may be a path to a TIFF (preferred for the real 5 GB file) or an
    in-memory (T, H, W) array (written to a temp TIFF first). suite2p does its
    OWN motion correction, so pass the RAW movie here, not a pre-registered one.

    `fs` = frame rate (Hz). `tau` = GCaMP decay time (s, ≈0.7-1.5 by indicator).
    `diameter` (px, optional) helps detection. Extra kwargs go into the suite2p
    `settings` dict. Returns F, Fneu, spks, stat, iscell, ops, dims, footprints.

    Note: suite2p 1.x replaced the old `run_s2p(ops, db)` / `default_ops()` API
    with `run_s2p(db, settings)` — input files + nplanes/nchannels live in `db`,
    while fs/tau/diameter live in `settings`.
    """
    import tifffile
    from suite2p import run_s2p

    os.makedirs(save_path, exist_ok=True)
    if isinstance(data, np.ndarray):
        mov = data
        if np.issubdtype(mov.dtype, np.floating):       # suite2p wants integer pixels
            mov = mov - mov.min()
            mov = (mov / (mov.max() + 1e-8) * 60000).astype(np.uint16)
        input_dir = os.path.join(save_path, "input")   # keep inputs out of the output folder
        os.makedirs(input_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(input_dir, "movie.tif"), mov)
        data_path, file_list = [input_dir], ["movie.tif"]
    else:
        data_path = [os.path.dirname(os.path.abspath(str(data)))]
        file_list = [os.path.basename(str(data))]

    db = dict(data_path=data_path, file_list=file_list, save_path0=save_path,
              nplanes=1, nchannels=1, input_format="tif")
    settings = dict(fs=fs, tau=tau)
    if diameter is not None:
        settings["diameter"] = [diameter, diameter] if np.isscalar(diameter) else list(diameter)
    settings.update(settings_kw)

    run_s2p(db=db, settings=settings)
    return load_suite2p_output(os.path.join(save_path, "suite2p", "plane0"))


def load_suite2p_output(plane_dir: str) -> dict:
    """Load suite2p's .npy outputs from a plane folder into a dict."""
    def L(name):
        return np.load(os.path.join(plane_dir, name), allow_pickle=True)

    ops = L("ops.npy").item()
    stat = L("stat.npy")
    dims = (ops["Ly"], ops["Lx"])
    return dict(
        F=L("F.npy"), Fneu=L("Fneu.npy"), spks=L("spks.npy"),
        stat=stat, iscell=L("iscell.npy"), ops=ops, dims=dims,
        footprints=footprints_from_stat(stat, dims),
    )


def footprints_from_stat(stat, dims) -> np.ndarray:
    """Build (N, H, W) footprint images from suite2p's per-ROI `stat` records."""
    fps = np.zeros((len(stat), *dims), np.float32)
    for i, s in enumerate(stat):
        fps[i, s["ypix"], s["xpix"]] = s["lam"]
    return fps
