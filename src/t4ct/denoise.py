"""Video denoising for Ca2+ imaging.

Classical baselines (run anywhere, no GPU) help segmentation and are a good
sanity check. The advanced milestone is deep self-supervised denoising:

- DeepCAD-RT  — Li et al., Nat. Methods 2021 (s41592-021-01285-2)
- DeepInterp. — Lecoq et al., Nat. Methods 2021 (s41592-021-01225-0)

Those need a GPU (the T4) and their own packages; see `deepcad_denoise` below
and the notebook's "advanced" section.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Classical baselines
# --------------------------------------------------------------------------- #
def temporal_average(mov: np.ndarray, k: int = 3) -> np.ndarray:
    """Moving average over time (window k). Cheap temporal denoise."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(np.asarray(mov, np.float32), size=k, axis=0, mode="nearest")


def gaussian_denoise(mov: np.ndarray, sigma=(0.0, 1.0, 1.0)) -> np.ndarray:
    """Spatial (and/or temporal) Gaussian smoothing. sigma is (t, y, x)."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(mov, np.float32), sigma=sigma)


def pca_denoise(mov: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Low-rank (PCA) temporal denoise — calcium movies are highly compressible,
    so keeping the top components removes a lot of noise while preserving cells."""
    from sklearn.decomposition import PCA
    mov = np.asarray(mov, np.float32)
    T = mov.shape[0]
    flat = mov.reshape(T, -1)
    pca = PCA(n_components=min(n_components, T - 1), svd_solver="randomized")
    rec = pca.inverse_transform(pca.fit_transform(flat))
    return rec.reshape(mov.shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# Metrics (when you have a clean reference, e.g. the synthetic ground truth)
# --------------------------------------------------------------------------- #
def psnr(clean: np.ndarray, test: np.ndarray) -> float:
    from skimage.metrics import peak_signal_noise_ratio
    clean, test = _match(clean), _match(test)
    return float(peak_signal_noise_ratio(clean, test, data_range=1.0))


def _match(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


# --------------------------------------------------------------------------- #
# Deep learning (advanced milestone) — optional, GPU
# --------------------------------------------------------------------------- #
def deepcad_denoise(mov: np.ndarray, **kw):
    """Thin wrapper around DeepCAD-RT (`pip install deepcad`).

    Self-supervised, so it trains on the noisy movie itself — no clean targets
    needed. Raises an informative error if the package isn't installed.
    """
    try:
        import deepcad  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "DeepCAD not installed. In Colab run `!pip install deepcad` (needs the "
            "T4 GPU). See https://github.com/cabooster/DeepCAD-RT for the API; this "
            "stub is a placeholder for the advanced milestone."
        ) from e
    raise NotImplementedError(
        "Wire up DeepCAD-RT's training/inference here once installed — see the repo."
    )
