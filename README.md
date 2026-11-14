# YOLOX training in C++ (LibTorch)

A C++17 / LibTorch / OpenCV port of the training pipeline in
[kyzira/YOLOX-Training-with-COCO](https://github.com/kyzira/YOLOX-Training-with-COCO)
(upstream Megvii YOLOX): fine-tune YOLOX on your own COCO-format dataset, evaluate COCO mAP
after every epoch and keep `best_ckpt.pth`.

The checkpoints use the same format in both directions:

* the official `yolox_*.pth` weights load by name, because the C++ modules use the same
  parameter names as the Python ones (`backbone.backbone.dark2.0.conv.weight`, …);
* checkpoints written by the C++ trainer are plain `torch.load`-able dicts
  (`{"model", "start_epoch", "best_ap", "curr_ap"}`), so the repo's `tools/export_onnx.py`
  and the ONNX inference demo work on them without changes.

## What was ported

| Python (yolox/…) | C++ (src/…) |
|---|---|
| `models/network_blocks.py`, `darknet.py` (CSPDarknet), `yolo_pafpn.py`, `yolox.py` | `models/network_blocks.*`, `models/yolo_pafpn.*`, `models/yolox.*` |
| `models/yolo_head.py` (decoupled head, **SimOTA**), `models/losses.py` (IoU loss) | `models/yolo_head.*` |
| `data/datasets/coco.py`, `mosaicdetection.py` (mosaic + mixup) | `data/coco_dataset.*` |
| `data/data_augment.py` (HSV, flip, random affine, preproc, TrainTransform) | `data/data_augment.*` |
| DataLoader + InfiniteSampler + prefetcher | `data/dataloader.*` (worker threads, pinned memory) |
| `exp/yolox_base.py`, `exps/default/*.py` | `exp/exp.*` + `exps/*.cfg` |
| `core/trainer.py` | `core/trainer.*` |
| `utils/lr_scheduler.py`, `utils/ema.py`, `utils/checkpoint.py`, `utils/boxes.py` | `core/lr_scheduler.hpp`, `core/ema.hpp`, `core/checkpoint.*`, `utils/boxes.*` |
| `evaluators/coco_evaluator.py` + pycocotools `COCOeval` | `core/evaluator.*` + `data/coco_api.*` |
| `tools/train.py`, `tools/eval.py`, `tools/demo.py` (image) | `tools/train.cpp`, `tools/eval.cpp`, `tools/demo.cpp` |

The trainer keeps all of the Python training behaviour:

* SGD (nesterov, momentum 0.9) with 3 parameter groups: BN weights with no decay, conv weights
  with decay 5e-4, and biases.
* `yoloxwarmcos` learning rate: quadratic warm-up, then cosine, with lr = `basic_lr_per_img * batch`.
* EMA with decay 0.9998 and ramp-up.
* Multiscale training: a random size every 10 iterations, applied with bilinear resize on the GPU.
* For the last `no_aug_epochs`: mosaic is turned off, the L1 loss is added and the model is
  evaluated every epoch.
* Checkpoints: `latest`, `last_epoch`, `last_mosaic_epoch`, `epoch_N` and `best_ckpt.pth`.
* `--resume`, `--fp16` (CUDA autocast plus a C++ GradScaler) and `--cache` (RAM).

## Requirements

You need CMake ≥ 3.18, a C++17 compiler, LibTorch ≥ 2.0 (a CUDA build for GPU training) and
OpenCV ≥ 4 (core, imgproc, imgcodecs). JSON parsing uses a small built-in parser, so no other
libraries are needed.

### Ubuntu packages

Ubuntu 22.04 and 24.04 work. On 20.04 the stock CMake (3.16) is too old: install a newer one
with `pip install cmake` or from Kitware's apt repository.

**1. Build tools and OpenCV**

```bash
sudo apt update
sudo apt install -y build-essential cmake git unzip wget libopencv-dev
```

| Package | Why |
|---|---|
| `build-essential` | GCC/G++ with C++17 support (GCC 9 or newer) |
| `cmake` | Build system (3.18 or newer) |
| `libopencv-dev` | OpenCV headers and libraries for image loading and augmentation |
| `git`, `unzip`, `wget` | Cloning the repo and downloading and unpacking LibTorch |

**2. LibTorch (not available from apt)**

Download the C++ build from <https://pytorch.org/get-started/locally/>. Choose
**LibTorch → C++/Java → Linux**, choose either CUDA or CPU, and pick the **cxx11 ABI** zip.

```bash
wget -O libtorch.zip "<URL from pytorch.org>"
unzip libtorch.zip -d ~/          # -> ~/libtorch
```

**3. NVIDIA GPU training**

- **NVIDIA driver:** `sudo ubuntu-drivers install` (or e.g. `sudo apt install nvidia-driver-550`).
  Check that it works with `nvidia-smi`.
- **CUDA toolkit:** the CMake files of a CUDA build of LibTorch look for the toolkit at build
  time. Its version should match the CUDA version of the LibTorch zip (for example `cu121`
  needs CUDA 12.1). Ubuntu's `nvidia-cuda-toolkit` package is often an older version, so
  install the matching version from NVIDIA's apt repository:
  <https://developer.nvidia.com/cuda-downloads>.
- cuDNN is included in the LibTorch zip, so you don't need to install it separately.

For CPU-only use (debugging), skip this step and download the CPU LibTorch zip.

**4. Python (optional)**

Python is only needed for `tools/convert_pth.py` and for ONNX export with the original Python
repo:

```bash
sudo apt install -y python3 python3-pip python3-venv
python3 -m venv ~/yolox-venv && source ~/yolox-venv/bin/activate
pip install torch            # the CPU build is enough for converting weights
```

## Build

```bash
git clone https://github.com/detu/yolox.cpp && cd yolox.cpp
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
#   or, with PyTorch installed via pip:
#   cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)") ..
cmake --build . -j"$(nproc)"
```

This builds `yolox_train`, `yolox_eval`, `yolox_demo` and `test_coco_eval`.

If the programs can't find the LibTorch libraries at runtime, add them to the library path:

```bash
export LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH
```

## Dataset

The dataset layout is the same as in the guide:

```
datasets/COCO/
├── annotations/instances_train2017.json
├── annotations/instances_val2017.json
├── train2017/*.jpg
└── val2017/*.jpg
```

You can change the folder names with `train_name` / `val_name` in the exp file.

## Exp file

Python exp files are code. In this port an exp file is a list of `key = value` lines (see
`exps/yolox_m.cfg`). Its content mirrors the guide's `exps/yolox_m.py`:

```ini
model = yolox-m          # depth 0.67, width 0.75   (yolox-nano|tiny|s|m|l|x)
data_dir = datasets/COCO
train_ann = instances_train2017.json
val_ann = instances_val2017.json
num_classes = 6
max_epoch = 50
data_num_workers = 4
eval_interval = 1
```

Every attribute of `yolox_base.py` can be set this way: `input_size = 640, 640`,
`mosaic_scale = 0.1, 2`, `enable_mixup = False`, `basic_lr_per_img = 0.01 / 64.0`, and so on.
You can also override them from the command line, like `exp.merge(opts)`:
`... max_epoch 100 test_conf 0.001`.

## Train

1. Download `yolox_m.pth` (or another size) from the
   [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0).
2. If the C++ loader can't read the downloaded file, convert it once. The official files contain
   Python-only optimizer/amp state:

   ```bash
   python tools/convert_pth.py yolox_m.pth yolox_m_cpp.pth
   ```

3. Start training. The command is the C++ version of
   `python tools/train.py -f exps/yolox_m.py -d 1 -b 9 --fp16 -o -c yolox_m.pth`:

   ```bash
   ./build/yolox_train -f exps/yolox_m.cfg -d 1 -b 9 --fp16 -o -c yolox_m_cpp.pth
   ```

   * `-c` fine-tunes from a checkpoint. The class layers (`head.cls_preds.*`) have a different
     shape when `num_classes != 80`, so they are skipped with a warning, just like in Python.
   * `--resume [-c ckpt] [-e start_epoch]` continues a run. The optimizer state is stored next
     to each checkpoint as `<name>_optim.pt`.
   * `--device -1` trains on the CPU. This is only useful for debugging.

Outputs go to `YOLOX_outputs/<exp_name>/` (`train_log.txt`, `*_ckpt.pth`, `best_ckpt.pth`).

## Evaluate / demo

```bash
./build/yolox_eval -f exps/yolox_m.cfg -c YOLOX_outputs/yolox_m/best_ckpt.pth -b 8 --conf 0.001
./build/yolox_demo -f exps/yolox_m.cfg -c YOLOX_outputs/yolox_m/best_ckpt.pth \
                   --path img.jpg --conf 0.3 --out vis.jpg --classes a,b,c,d,e,f
```

## Export to ONNX

LibTorch has no ONNX exporter, so export with the original Python tool. It reads the C++
checkpoint directly:

```bash
python tools/export_onnx.py --output-name trained_yolox.onnx -f exps/yolox_m.py \
       -c YOLOX_outputs/yolox_m/best_ckpt.pth
```

## Differences from the Python repo

* **Single GPU only.** DDP and multi-machine training (`--dist-url`, `--num_machines`) are not
  ported. `-d N` with N > 1 prints a warning.
* **Logging** goes to stdout and `train_log.txt`. There is no TensorBoard, W&B or MLflow.
* **Not ported:** the YOLOv3/Darknet53 exp (`yolov3.py`), VOC datasets, disk caching
  (`--cache disk`), the `multistep` and `yoloxsemiwarmcos` schedulers, the assignment
  visualiser, TensorRT eval, and `--fuse`.
* **Random streams:** augmentation uses `std::mt19937_64` instead of Python `random`/`numpy`.
  The distributions and code paths are the same, but samples are not bit-identical to a Python
  run.
* **fp16:** inputs and targets stay in fp32 and autocast casts the convolutions. SimOTA cost
  computation always runs in fp32, as in Python. The class cost is computed in closed form
  instead of materialising `[num_gt, num_anchors, num_classes]` tensors (same values, less
  memory).
* **Category names** come from the annotation file. `class_ids` are the sorted category ids,
  as in `coco.py`.

## Tests

`test_coco_eval GT.json DETS.json` runs the built-in COCOeval. On randomly generated datasets
it prints exactly the same `summarize()` table, 12 stats and per-class APs as pycocotools,
including crowd boxes, score ties and more than 100 detections per image.
