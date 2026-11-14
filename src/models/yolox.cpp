#include "models/yolox.hpp"

namespace yolox {

YOLOXImpl::YOLOXImpl(const ModelConfig& c) : cfg(c) {
  const Act act = act_from_string(cfg.act);
  const std::array<int64_t, 3> in_channels{256, 512, 1024};
  backbone = register_module(
      "backbone", YOLOPAFPN(cfg.depth, cfg.width, in_channels, cfg.depthwise, act));
  head = register_module("head", YOLOXHead(cfg.num_classes, cfg.width,
                                           std::array<int64_t, 3>{8, 16, 32},
                                           in_channels, act, cfg.depthwise));
}

torch::Tensor YOLOXImpl::forward(const torch::Tensor& x) {
  return head->forward(backbone->forward(x));
}

LossOutput YOLOXImpl::forward_train(const torch::Tensor& x, const torch::Tensor& targets) {
  return head->forward_train(backbone->forward(x), targets);
}

YOLOX build_model(const ModelConfig& cfg) {
  YOLOX model(cfg);
  model->head->initialize_biases(1e-2);
  model->train();
  return model;
}

}  // namespace yolox
