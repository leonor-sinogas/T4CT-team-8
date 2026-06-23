# Troubleshooting

The real-world snags we hit running this on Colab + locally, and the fixes.
suite2p-specific GUI/export issues are in [local-gui-and-export.md](local-gui-and-export.md);
DeepInterpolation issues are in [deepinterpolation.md](deepinterpolation.md).

## Colab

### suite2p needs a one-time runtime restart after install
suite2p pins a `numpy`/`numba` combo different from Colab's default (numba needs
`numpy <= 2.2`). The install cell installs, then auto-restarts the runtime — just
**Run all again** when it returns (it skips the install the second time).

### Installing deepinterpolation + suite2p together fails
`pip install deepinterpolation suite2p` in one command can't be resolved
(TensorFlow vs numba disagree on numpy). Install them **separately**, and install
deepinterpolation from **GitHub** with `--no-deps` (it's pure Python and reuses
Colab's TensorFlow):
```python
!pip install --no-deps git+https://github.com/AllenInstitute/deepinterpolation.git
!pip install nibabel        # deepinterpolation dep not on Colab
!pip install suite2p
```

### `ModuleNotFoundError: No module named 's3fs'`
You installed the **PyPI** deepinterpolation (0.2.0), which hard-imports `s3fs`
(doesn't install cleanly on Colab). Use the **GitHub** version instead (above),
then restart the runtime.

### `ValueError: Argument(s) not recognized: {'lr': ...}` (DeepInterpolation)
The pretrained `.h5` has an old Keras-2 optimizer (`lr`/`decay`) that Keras 3
rejects. Inference only needs the weights — load with `compile=False`:
```python
import deepinterpolation.inference_collection as _ic
_orig = _ic.load_model
_ic.load_model = lambda path, **kw: _orig(path, **{**kw, "compile": False})
```
(The notebook does this in its denoise cell.)

### suite2p programmatic API (1.x)
suite2p ≥ 1.0 replaced `run_s2p(ops, db)` + `default_ops()` with
`run_s2p(db, settings)` — input files + `nplanes`/`nchannels` go in `db`,
`fs`/`tau`/`diameter` go in `settings`. `t4ct.segment.run_suite2p` already uses
the new API.

### Accessing a shared folder from Drive
"Shared with me" has no stable path. In Drive (web): right-click the shared
folder → **Organize → Add shortcut to Drive**, then mount and use
`/content/drive/MyDrive/<shortcut name>/...`. (A true Workspace *Shared drive*
appears at `/content/drive/Shared drives/<name>/`.)

### Kernel runs out of RAM
The correlation-image baseline and loading a full movie into memory are
RAM-heavy. Subsample frames (`mov[::2]`), crop the FOV, or use
`data.load_tiff(path, memmap=True)`. Or just let suite2p read the TIFF directly.

## Local (VS Code / Jupyter)

### `Running cells … requires the ipykernel package`
You selected the wrong interpreter (e.g. system `python3.12`/`3.13`). Select the
project kernel **"Python (T4CT suite2p)"**, or point VS Code at
`…/embrio-T4CT/suite2p_venv/bin/python` via *Select Kernel → Python Environments
→ Enter interpreter path*. (System Python 3.13 is unsupported by suite2p; the
venv is 3.12.)

### The kernel doesn't show up
The venv lives in `embrio-T4CT/` (one level above the repo), so VS Code's
auto-discovery misses it when you open only `T4CT-team-8/`. Either open
`embrio-T4CT/` as the workspace, or use *Enter interpreter path* (above).

### Colab notebooks have cells that don't work locally
The notebooks target Colab. `!nvidia-smi`, the install/auto-restart cell,
`/content/...` paths, `from google.colab import drive`, and the GPU/TensorFlow
steps only work on Colab. Use the local kernel for inspecting outputs and the
lighter `t4ct` code; run the real pipeline on Colab (T4).

### Harmless warnings (not errors)
`cellpose.vit: Could not import CPDINO …` and `pynwb not installed …` are
optional-feature notices — ignore them. (`pip install pynwb` removes the second.)
