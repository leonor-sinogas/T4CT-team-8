# DeepInterpolation denoising (2-photon)

[DeepInterpolation](https://github.com/AllenInstitute/deepinterpolation) (Allen
Institute; Lecoq et al., *Nat. Methods* 2021) is **self-supervised** denoising
for systems-neuroscience movies. It predicts each frame from its neighbours
(30 before + 30 after by default); independent shot noise can't be predicted, so
it gets removed — no clean ground truth required.

> **Run it on Colab (T4).** It needs TensorFlow + a GPU. Colab has both; local
> M2/CPU inference works but is very slow (the upstream example says so).

## What was fetched (in the `embrio-T4CT` folder)

The repo and a pretrained 2-photon model are cloned **outside** the git repo
(they're large and not committed):

```
embrio-T4CT/
├── deepinterpolation/                     # cloned Allen repo (package + examples + sample data)
│   ├── examples/example_tiny_ophys_inference.py   # the 2p inference example we adapted
│   ├── examples/example_tiny_ophys_training.py    # self-supervised training on a TIFF
│   ├── sample_data/ophys_tiny_761605196.tif       # a sample 2p movie (50 MB)
│   └── pretrained_models/ai93/                     # downloaded Ai93 2p models
│       └── 2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5   # 120 MB
└── T4CT-team-8/                            # this repo (committed)
    ├── notebooks/deepinterp_denoise.ipynb # Colab runner (recommended)
    └── scripts/deepinterp_denoise.py      # CLI: infer (TIFF->H5) + to-tif (H5->TIFF)
```

The Ai93 model came from Allen's
[Dropbox](https://www.dropbox.com/sh/vwxf1uq2j60uj9o/AAC9BQI1bdfmAL3OFO0lmVb1a?dl=0)
(`deep_interpolation_ai93_v1_1.zip`, ~185 MB). It's the exact file the upstream
ophys example references.

## How to run

### Colab (recommended)
Open **`notebooks/deepinterp_denoise.ipynb`** → T4 GPU → Run all. It's the whole
comparison in one place: installs DeepInterpolation **+ suite2p**, downloads the
model, denoises a movie, then runs **suite2p on the raw vs the denoised movie**
and compares **# cells and trace SNR**. To use your data, mount Drive and set
`TIF` (cell 4); keep `end_frame=-1` (cell 5) for the full movie.

> The first run installs suite2p, which forces a **one-time runtime restart** —
> just Run all again when it returns.

### CLI (any machine with deepinterpolation + TF installed)
```bash
# install from GitHub, not PyPI — the 0.2.0 PyPI release hard-imports s3fs
pip install --no-deps git+https://github.com/AllenInstitute/deepinterpolation.git
pip install tifffile h5py nibabel

python scripts/deepinterp_denoise.py infer \
    --tif   /path/recording.tif \
    --model /path/to/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5 \
    --out   denoised.h5 --end -1

python scripts/deepinterp_denoise.py to-tif --h5 denoised.h5 --out denoised.tif
```

The denoised HDF5 has a dataset `data` of shape `(frames, H, W)`. Output has
`2 × pre_post` (=60) fewer frames than the input — frame `i` of the output
corresponds to frame `30 + i` of the input.

## Pretrained vs. training your own

The Ai93 model was trained on Allen data (**GCaMP6, 512×512, 30 Hz, ~400 µm FOV**).
Our recording is 30 fps mouse cortex — close enough to try directly. If results
look poor (different indicator/zoom/resolution), **train your own** model — it's
self-supervised, so it learns from your noisy movie with no labels:

```bash
# adapt examples/example_tiny_ophys_training.py: set train_path to your TIFF,
# raise end_frame / nb_times_through_data, set nb_gpus=1, then run on the T4.
```

A few epochs on a subset is enough for a hackathon demo.

## Troubleshooting

- **`ModuleNotFoundError: No module named 's3fs'` (cell 5).** You installed the
  PyPI release (0.2.0), which hard-imports `s3fs`. Install from GitHub instead:
  `pip uninstall -y deepinterpolation && pip install --no-deps git+https://github.com/AllenInstitute/deepinterpolation.git`,
  then restart the runtime.
- **`ValueError: Argument(s) not recognized: {'lr': ...}` (cell 5, in `load_model`).**
  The pretrained `.h5` carries an old Keras-2 optimizer (`lr`/`decay`) that Keras 3
  (current Colab TF) rejects. Inference only needs the weights — load with
  `compile=False`. The notebook does this by patching `load_model`:
  ```python
  import deepinterpolation.inference_collection as _ic
  _orig = _ic.load_model
  _ic.load_model = lambda path, **kw: _orig(path, **{**kw, "compile": False})
  ```
- **Other `load_model` deserialization errors.** Try the newer 2021 models in
  `pretrained_models/ai93/` (e.g. `..._feat_32_power_2_depth_4_unet_True-0100-0.5733.h5`),
  or use legacy Keras 2: `pip install tf-keras` and set
  `os.environ["TF_USE_LEGACY_KERAS"]="1"` **before** importing TensorFlow (then restart).
- **Out of memory on a big movie.** Process in chunks with `--start/--end`
  (CLI) or `start_frame/end_frame` (notebook), then concatenate.
- **Wrong frame count.** Expected — see the 60-frame edge loss above.

## After denoising

Save `denoised.tif` to Drive and run it through the suite2p pipeline
(`notebooks/T4CT_demo.ipynb`). Compare ROI count and ΔF/F SNR against the raw
movie — that delta is the *intermediate* milestone in the project brief.
