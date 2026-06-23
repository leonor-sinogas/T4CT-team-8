"""Extract and normalise Ca2+ activity traces from neuron footprints."""
from __future__ import annotations

import numpy as np


def extract_traces(mov: np.ndarray, footprints: np.ndarray) -> np.ndarray:
    """Weighted-mean fluorescence per ROI over time -> (N, T).

    (Use this with the `correlation_segment` baseline. suite2p already returns
    its own traces in `out["F"]`.)
    """
    mov = np.asarray(mov, np.float32)
    T = mov.shape[0]
    flat = mov.reshape(T, -1)                      # (T, P)
    F = footprints.reshape(len(footprints), -1)    # (N, P)
    weights = F.sum(1, keepdims=True) + 1e-8
    return (F @ flat.T) / weights                  # (N, T)


def dff(F: np.ndarray, percentile: float = 10.0) -> np.ndarray:
    """ΔF/F0 with a per-cell percentile baseline. F may be neuropil-corrected
    (e.g. `out["F"] - 0.7 * out["Fneu"]`)."""
    F = np.asarray(F, np.float32)
    F0 = np.percentile(F, percentile, axis=1, keepdims=True)
    return (F - F0) / (np.abs(F0) + 1e-8)


def neuropil_correct(F: np.ndarray, Fneu: np.ndarray, coef: float = 0.7) -> np.ndarray:
    """Standard suite2p neuropil subtraction: F_corr = F - coef * Fneu."""
    return np.asarray(F, np.float32) - coef * np.asarray(Fneu, np.float32)
