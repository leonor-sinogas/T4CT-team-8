"""Video IO, summary images, and a synthetic Ca2+ movie generator.

A two-photon recording is a (T, H, W) stack: T frames of an H x W field of view.
Everything here speaks that shape. tifffile is imported lazily so `import t4ct`
works without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# IO  (the real recording is a multi-page TIFF, ~5 GB / 10000 frames)
# --------------------------------------------------------------------------- #
def load_tiff(path: str | Path, n_frames: Optional[int] = None,
              memmap: bool = False) -> np.ndarray:
    """Load a multi-page TIFF as a (T, H, W) array.

    memmap=True returns a disk-backed array (use this for the 5 GB file so you
    don't blow up Colab RAM). n_frames loads only the first N pages.
    """
    import tifffile
    if memmap:
        return tifffile.memmap(str(path))
    key = range(n_frames) if n_frames else None
    return tifffile.imread(str(path), key=key)


def save_tiff(path: str | Path, mov: np.ndarray) -> None:
    import tifffile
    tifffile.imwrite(str(path), np.asarray(mov))


# --------------------------------------------------------------------------- #
# Summary images  (cheap views over the whole movie)
# --------------------------------------------------------------------------- #
def mean_image(mov: np.ndarray) -> np.ndarray:
    return np.asarray(mov, np.float32).mean(0)


def max_image(mov: np.ndarray) -> np.ndarray:
    return np.asarray(mov, np.float32).max(0)


def correlation_image(mov: np.ndarray) -> np.ndarray:
    """Local correlation image: each pixel's mean temporal correlation with its
    4 neighbours. Active neurons light up — the go-to map for spotting cells."""
    m = np.asarray(mov, np.float32)
    m = m - m.mean(0, keepdims=True)
    std = np.sqrt((m ** 2).mean(0)) + 1e-8
    norm = m / std
    corr = np.zeros(m.shape[1:], np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        corr += (norm * np.roll(norm, (dy, dx), axis=(1, 2))).mean(0)
    return corr / 4.0


# --------------------------------------------------------------------------- #
# Synthetic movie — develop the whole pipeline before the real data lands.
# --------------------------------------------------------------------------- #
def synthetic_movie(n_frames: int = 600, size: int = 256, n_cells: int = 80,
                    fps: float = 30.0, tau: float = 0.8, motion: bool = False,
                    noise: float = 1.0, seed: int = 0
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a fake two-photon movie with known ground truth.

    Returns (movie[T,H,W], footprints[N,H,W], traces[N,T]). Neurons are Gaussian
    blobs; traces are Poisson spikes convolved with an exponential Ca2+ decay.
    With motion=True a random-walk shift is applied so you can test correction.
    """
    from scipy.ndimage import gaussian_filter, shift as nd_shift

    rng = np.random.default_rng(seed)
    H = W = size

    # Spatial footprints: Gaussian blobs at random locations.
    ys = rng.integers(8, H - 8, n_cells)
    xs = rng.integers(8, W - 8, n_cells)
    radii = rng.uniform(2.5, 5.0, n_cells)
    yy, xx = np.mgrid[0:H, 0:W]
    footprints = np.zeros((n_cells, H, W), np.float32)
    for i in range(n_cells):
        g = np.exp(-((yy - ys[i]) ** 2 + (xx - xs[i]) ** 2) / (2 * radii[i] ** 2))
        footprints[i] = (g / g.max()).astype(np.float32)

    # Temporal traces: sparse spikes -> exponential calcium kernel.
    decay = np.exp(-1.0 / (tau * fps))
    kernel = decay ** np.arange(int(tau * fps * 5) + 1)
    traces = np.zeros((n_cells, n_frames), np.float32)
    rates = rng.uniform(0.02, 0.15, n_cells)
    for i in range(n_cells):
        amp = rng.uniform(0.5, 2.0, n_frames) * (rng.random(n_frames) < rates[i])
        traces[i] = np.convolve(amp, kernel)[:n_frames]

    mov = np.tensordot(traces.T, footprints, axes=(1, 0)) + 0.2   # (T, H, W)
    mov = gaussian_filter(mov, sigma=(0, 0.6, 0.6))               # mild blur (PSF)

    if motion:
        walk = np.cumsum(rng.normal(0, 0.3, (n_frames, 2)), axis=0)
        walk -= walk.mean(0)
        for t in range(n_frames):
            mov[t] = nd_shift(mov[t], walk[t], order=1, mode="nearest")

    # Shot noise (Poisson) + read noise (Gaussian).
    mov = rng.poisson(np.clip(mov, 0, None) * 30) / 30.0
    mov = mov + rng.normal(0, noise * 0.05, mov.shape)
    return np.clip(mov, 0, None).astype(np.float32), footprints, traces
