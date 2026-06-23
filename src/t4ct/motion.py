"""Motion correction.

`register_rigid` is a dependency-light baseline (phase cross-correlation) good
for quick iteration and short clips. For the full 10000-frame recording prefer
suite2p's built-in registration (it runs as part of `segment.run_suite2p`,
which is faster and supports non-rigid correction).

Reference: Pnevmatikakis & Giovannucci, "NoRMCorre", J. Neurosci. Methods 2017.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def reference_image(mov: np.ndarray, n: int = 500) -> np.ndarray:
    """Mean of the first `n` frames — a simple registration template."""
    mov = np.asarray(mov, np.float32)
    return mov[: min(n, len(mov))].mean(0)


def register_rigid(mov: np.ndarray, reference: Optional[np.ndarray] = None,
                   upsample: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid (translation-only) motion correction.

    Returns (corrected_movie, shifts[T, 2]) where shifts are (dy, dx) per frame.
    Slow for very long movies (Python loop) — use suite2p for the full dataset.
    """
    from scipy.ndimage import shift as nd_shift
    from skimage.registration import phase_cross_correlation

    mov = np.asarray(mov, np.float32)
    if reference is None:
        reference = reference_image(mov)

    out = np.empty_like(mov)
    shifts = np.zeros((len(mov), 2), np.float32)
    for i, frame in enumerate(mov):
        s, _, _ = phase_cross_correlation(reference, frame, upsample_factor=upsample)
        shifts[i] = s
        out[i] = nd_shift(frame, s, order=1, mode="nearest")
    return out, shifts
