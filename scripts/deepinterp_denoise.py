#!/usr/bin/env python
"""Denoise a 2-photon TIFF movie with DeepInterpolation (Allen Institute).

DeepInterpolation is self-supervised denoising for systems-neuroscience movies
(Lecoq et al., Nat. Methods 2021). For each frame it predicts the frame from its
neighbours (30 before + 30 after by default); independent noise can't be
predicted, so it's removed.

Subcommands
-----------
  infer    TIFF  -> denoised HDF5   (runs the model; needs deepinterpolation + TF)
  to-tif   HDF5  -> TIFF            (convert the output to something viewable)

Recommended runtime: Google Colab with a T4 GPU (TensorFlow + CUDA ready). Local
CPU inference works but is very slow (the upstream example warns about this).

Examples
--------
  # denoise the whole movie (end=-1) with the pretrained Ai93 2p model
  python deepinterp_denoise.py infer \\
      --tif   /path/recording.tif \\
      --model /path/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5 \\
      --out   denoised.h5 --end -1

  # make it viewable
  python deepinterp_denoise.py to-tif --h5 denoised.h5 --out denoised.tif

Note: the output has 2*pre_post fewer frames than the input (30 lost at each end
by default) because edge frames lack enough neighbours.
"""
from __future__ import annotations

import argparse
import sys


def infer(args):
    from deepinterpolation.generator_collection import SingleTifGenerator
    from deepinterpolation.inference_collection import core_inference

    generator_param = {
        "pre_post_frame": args.pre_post,
        "pre_post_omission": 0,
        "steps_per_epoch": -1,
        "train_path": args.tif,          # the input movie (the generator name is historical)
        "batch_size": args.batch_size,
        "start_frame": args.start,
        "end_frame": args.end,           # -1 = until the end
        "randomize": 0,                  # keep temporal order for inference
    }
    inference_param = {"model_path": args.model, "output_file": args.out}

    print(f"input : {args.tif}")
    print(f"model : {args.model}")
    print(f"frames: {args.start}..{args.end}  pre_post={args.pre_post}  batch={args.batch_size}")
    gen = SingleTifGenerator(generator_param)
    core_inference(inference_param, gen).run()
    print(f"✅ denoised movie -> {args.out}  (HDF5 dataset 'data')")


def to_tif(args):
    import h5py
    import numpy as np
    import tifffile

    with h5py.File(args.h5, "r") as f:
        key = "data" if "data" in f else list(f.keys())[0]
        data = np.squeeze(f[key][()])
    tifffile.imwrite(args.out, data.astype("float32"))
    print(f"{args.h5}['{key}'] {data.shape} -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer", help="denoise a TIFF -> HDF5")
    p.add_argument("--tif", required=True, help="input 3D TIFF movie (frames, H, W)")
    p.add_argument("--model", required=True, help="pretrained DeepInterpolation .h5 model")
    p.add_argument("--out", default="denoised.h5", help="output HDF5 (default denoised.h5)")
    p.add_argument("--pre-post", type=int, default=30, dest="pre_post",
                   help="frames of context before/after each frame (default 30)")
    p.add_argument("--batch-size", type=int, default=5, dest="batch_size", help="batch size (default 5)")
    p.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    p.add_argument("--end", type=int, default=-1, help="last frame, -1 = end (default -1)")
    p.set_defaults(func=infer)

    p = sub.add_parser("to-tif", help="convert a denoised HDF5 -> TIFF")
    p.add_argument("--h5", required=True, help="HDF5 produced by `infer`")
    p.add_argument("--out", default="denoised.tif", help="output TIFF (default denoised.tif)")
    p.set_defaults(func=to_tif)

    args = ap.parse_args()
    try:
        args.func(args)
    except ImportError as e:
        sys.exit(f"\nMissing dependency: {e}\n"
                 f"Install with:  pip install --no-deps "
                 f"git+https://github.com/AllenInstitute/deepinterpolation.git\n"
                 f"               pip install tifffile h5py nibabel\n"
                 f"(Install from GitHub, not PyPI — the 0.2.0 release hard-imports s3fs.\n"
                 f" Best run on Colab with a T4 GPU — see docs/deepinterpolation.md)")


if __name__ == "__main__":
    main()
