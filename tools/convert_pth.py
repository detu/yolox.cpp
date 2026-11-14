#!/usr/bin/env python3
"""Re-save a Python YOLOX checkpoint so LibTorch's torch::pickle_load can read it.

The official yolox_*.pth files (and checkpoints written by the Python trainer)
contain an OrderedDict plus optimizer/amp state. The C++ loader only needs the
model weights, stored as a plain {name: tensor} dict in the zip format:

    python tools/convert_pth.py yolox_m.pth yolox_m_cpp.pth

Checkpoints written by the C++ trainer do not need converting; they already
load in Python with torch.load(path)["model"] (so tools/export_onnx.py works).
"""
import sys

import torch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    out = {"model": {k: v.contiguous() for k, v in sd.items() if torch.is_tensor(v)}}
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("start_epoch"), int):
            out["start_epoch"] = ckpt["start_epoch"]
        if isinstance(ckpt.get("best_ap"), (int, float)):
            out["best_ap"] = float(ckpt["best_ap"])
    torch.save(out, dst, _use_new_zipfile_serialization=True)
    print(f"wrote {dst} ({len(out['model'])} tensors)")


if __name__ == "__main__":
    main()
