// C++ equivalent of yolox/exp/yolox_base.py + exps/default/*.py
//
// Python exp files are code; here an experiment is a plain struct filled from
// (1) a model preset (-n yolox-m), (2) an optional "key = value" config file
// (-f exps/yolox_m.cfg) and (3) trailing "key value" overrides on the command
// line, exactly like `exp.merge(args.opts)`.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "models/yolox.hpp"

namespace yolox {

struct Exp {
  // ---------------- model config
  int num_classes = 80;
  double depth = 1.00;
  double width = 1.00;
  std::string act = "silu";
  bool depthwise = false;  // yolox-nano

  // ---------------- dataloader config
  int data_num_workers = 4;
  std::pair<int, int> input_size{640, 640};  // (height, width)
  int multiscale_range = 5;
  std::optional<std::pair<int, int>> random_size;  // overrides multiscale_range
  std::string data_dir = "datasets/COCO";
  std::string train_ann = "instances_train2017.json";
  std::string val_ann = "instances_val2017.json";
  std::string train_name = "train2017";  // image folder names
  std::string val_name = "val2017";

  // ---------------- transform config
  double mosaic_prob = 1.0;
  double mixup_prob = 1.0;
  double hsv_prob = 1.0;
  double flip_prob = 0.5;
  double degrees = 10.0;
  double translate = 0.1;
  std::pair<double, double> mosaic_scale{0.1, 2.0};
  bool enable_mixup = true;
  std::pair<double, double> mixup_scale{0.5, 1.5};
  double shear = 2.0;

  // ---------------- training config
  int warmup_epochs = 5;
  int max_epoch = 300;
  double warmup_lr = 0;
  double min_lr_ratio = 0.05;
  double basic_lr_per_img = 0.01 / 64.0;
  std::string scheduler = "yoloxwarmcos";
  int no_aug_epochs = 15;
  bool ema = true;
  double weight_decay = 5e-4;
  double momentum = 0.9;
  int print_interval = 10;
  int eval_interval = 10;
  bool save_history_ckpt = true;
  std::string exp_name = "yolox";
  std::string output_dir = "./YOLOX_outputs";
  std::optional<uint64_t> seed;

  // ---------------- testing config
  std::pair<int, int> test_size{640, 640};  // (height, width)
  double test_conf = 0.01;
  double nmsthre = 0.65;

  ModelConfig model_config() const;
  void apply_preset(const std::string& name, bool set_exp_name = true);  // yolox-nano/tiny/s/m/l/x
  void load_file(const std::string& path);     // key = value lines
  void set(const std::string& key, const std::string& value);
  void check() const;                          // check_exp_value()
  std::string to_string() const;
};

}  // namespace yolox
