"""T4CT — two-photon Ca2+ imaging analysis toolkit.

Pipeline: load video -> motion-correct -> denoise -> segment neuron footprints
-> extract Ca2+ traces. Built around suite2p, with lightweight baselines and a
synthetic-movie generator so the pipeline can be developed before the real
~5 GB recording is available.

Submodules import their heavy/optional dependencies lazily (inside functions),
so `import t4ct` works even if suite2p / tifffile aren't installed yet.

Typical use in the Colab notebook:

    from t4ct import data, motion, denoise, segment, traces, viz

    mov, true_fp, true_tr = data.synthetic_movie()      # or data.load_tiff(path)
    mov_mc, shifts = motion.register_rigid(mov)
    out = segment.run_suite2p(mov, save_path="/content/s2p")
    dff = traces.dff(out["F"] - 0.7 * out["Fneu"])
    viz.plot_footprints(out["footprints"], out["ops"]["meanImg"])
    viz.plot_traces(dff, fps=30)
"""

from . import data, motion, denoise, segment, traces, viz  # noqa: F401

__all__ = ["data", "motion", "denoise", "segment", "traces", "viz"]
__version__ = "0.1.0"
