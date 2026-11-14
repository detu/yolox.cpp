// C++/LibTorch port of yolox/models/yolox.py + Exp.get_model()
#pragma once

#include "models/yolo_head.hpp"
#include "models/yolo_pafpn.hpp"

namespace yolox {

struct ModelConfig {
  int64_t num_classes = 80;
  double depth = 1.0;
  double width = 1.0;
  bool depthwise = false;
  std::string act = "silu";
};

struct YOLOXImpl : torch::nn::Module {
  explicit YOLOXImpl(const ModelConfig& cfg);

  // Inference / eval: decoded predictions [B, A, 5 + C]
  torch::Tensor forward(const torch::Tensor& x);
  // Training: losses
  LossOutput forward_train(const torch::Tensor& x, const torch::Tensor& targets);

  ModelConfig cfg;
  YOLOPAFPN backbone{nullptr};
  YOLOXHead head{nullptr};
};
TORCH_MODULE(YOLOX);

// Equivalent of Exp.get_model(): builds the network, BN eps/momentum set in
// BaseConv, and head biases initialised with prior 1e-2.
YOLOX build_model(const ModelConfig& cfg);

}  // namespace yolox
