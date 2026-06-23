"""Plotting helpers for the notebook (matplotlib, preinstalled on Colab)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def _norm(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def show_summary(mov: np.ndarray):
    """Mean / max / correlation image side by side — the three standard views."""
    import matplotlib.pyplot as plt
    from .data import correlation_image, max_image, mean_image

    views = {"mean": mean_image(mov), "max": max_image(mov),
             "correlation": correlation_image(mov)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, img) in zip(axes, views.items()):
        ax.imshow(_norm(img), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def montage(mov: np.ndarray, n: int = 8, cmap: str = "gray"):
    """Show `n` evenly-spaced frames to eyeball the recording."""
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(mov) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4))
    for ax, i in zip(axes, idx):
        ax.imshow(_norm(mov[i]), cmap=cmap)
        ax.set_title(f"f{i}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_footprints(footprints: np.ndarray, background: np.ndarray,
                    title: str = "ROIs", thresh: float = 0.2):
    """Overlay coloured neuron footprints on a background (e.g. the mean image)."""
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb

    labels = np.zeros(np.asarray(background).shape, dtype=int)
    for i, fp in enumerate(footprints, start=1):
        peak = fp.max()
        labels[fp > thresh * peak] = i if peak > 0 else 0
    overlay = label2rgb(labels, image=_norm(background), bg_label=0, alpha=0.45)
    plt.figure(figsize=(7, 7))
    plt.imshow(np.clip(overlay, 0, 1))
    plt.title(f"{title} (n={len(footprints)})")
    plt.axis("off")
    plt.show()


def plot_traces(traces: np.ndarray, fps: float = 30.0, n: int = 50,
                spacing: Optional[float] = None, color: str = "k"):
    """Stacked Ca2+ traces over time (like the figures in the project brief)."""
    import matplotlib.pyplot as plt
    tr = np.asarray(traces)[:n]
    t = np.arange(tr.shape[1]) / fps
    if spacing is None:
        spacing = float(np.nanpercentile(tr, 99)) * 1.2 or 1.0
    plt.figure(figsize=(11, max(4, 0.18 * len(tr))))
    for i, row in enumerate(tr):
        plt.plot(t, row + i * spacing, color=color, lw=0.5)
    plt.xlabel("time (s)")
    plt.yticks([])
    plt.title(f"Ca2+ traces (n={len(tr)})")
    plt.tight_layout()
    plt.show()
