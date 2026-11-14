#include "models/network_blocks.hpp"

#include <stdexcept>

namespace yolox {

using torch::indexing::None;
using torch::indexing::Slice;

Act act_from_string(const std::string& name) {
  if (name == "silu") return Act::SiLU;
  if (name == "relu") return Act::ReLU;
  if (name == "lrelu") return Act::LReLU;
  throw std::invalid_argument("Unsupported act type: " + name);
}

torch::Tensor apply_act(const torch::Tensor& x, Act act) {
  switch (act) {
    case Act::SiLU:
      return torch::silu(x);
    case Act::ReLU:
      return torch::relu(x);
    case Act::LReLU:
      return torch::leaky_relu(x, 0.1);
  }
  return x;
}

// ---------------------------------------------------------------- BaseConv
BaseConvImpl::BaseConvImpl(int64_t in_channels, int64_t out_channels,
                           int64_t ksize, int64_t stride, int64_t groups,
                           bool bias, Act act)
    : act_(act) {
  const int64_t pad = (ksize - 1) / 2;
  conv = register_module(
      "conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,
                                                         out_channels, ksize)
                                    .stride(stride)
                                    .padding(pad)
                                    .groups(groups)
                                    .bias(bias)));
  // Exp.get_model() -> init_yolo(): BN eps=1e-3, momentum=0.03
  bn = register_module(
      "bn", torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(out_channels).eps(1e-3).momentum(
                    0.03)));
}

torch::Tensor BaseConvImpl::forward(const torch::Tensor& x) {
  return apply_act(bn->forward(conv->forward(x)), act_);
}

// ---------------------------------------------------------------- DWConv
DWConvImpl::DWConvImpl(int64_t in_channels, int64_t out_channels,
                       int64_t ksize, int64_t stride, Act act) {
  dconv = register_module(
      "dconv", BaseConv(in_channels, in_channels, ksize, stride,
                        /*groups=*/in_channels, false, act));
  pconv = register_module(
      "pconv", BaseConv(in_channels, out_channels, 1, 1, 1, false, act));
}

torch::Tensor DWConvImpl::forward(const torch::Tensor& x) {
  return pconv->forward(dconv->forward(x));
}

torch::nn::AnyModule make_conv(bool depthwise, int64_t in_channels,
                               int64_t out_channels, int64_t ksize,
                               int64_t stride, Act act) {
  if (depthwise) {
    return torch::nn::AnyModule(
        DWConv(in_channels, out_channels, ksize, stride, act));
  }
  return torch::nn::AnyModule(
      BaseConv(in_channels, out_channels, ksize, stride, 1, false, act));
}

// ---------------------------------------------------------------- Bottleneck
BottleneckImpl::BottleneckImpl(int64_t in_channels, int64_t out_channels,
                               bool shortcut, double expansion, bool depthwise,
                               Act act) {
  const auto hidden = static_cast<int64_t>(out_channels * expansion);
  conv1 = register_module("conv1",
                          BaseConv(in_channels, hidden, 1, 1, 1, false, act));
  conv2 = make_conv(depthwise, hidden, out_channels, 3, 1, act);
  register_module("conv2", conv2.ptr());
  use_add = shortcut && in_channels == out_channels;
}

torch::Tensor BottleneckImpl::forward(const torch::Tensor& x) {
  auto y = conv2.forward(conv1->forward(x));
  if (use_add) y = y + x;
  return y;
}

// ---------------------------------------------------------------- SPPBottleneck
SPPBottleneckImpl::SPPBottleneckImpl(int64_t in_channels, int64_t out_channels,
                                     std::vector<int64_t> kernel_sizes, Act act)
    : kernel_sizes_(std::move(kernel_sizes)) {
  const int64_t hidden = in_channels / 2;
  conv1 = register_module("conv1",
                          BaseConv(in_channels, hidden, 1, 1, 1, false, act));
  const int64_t conv2_channels =
      hidden * static_cast<int64_t>(kernel_sizes_.size() + 1);
  conv2 = register_module(
      "conv2", BaseConv(conv2_channels, out_channels, 1, 1, 1, false, act));
}

torch::Tensor SPPBottleneckImpl::forward(const torch::Tensor& input) {
  auto x = conv1->forward(input);
  std::vector<torch::Tensor> feats{x};
  for (auto ks : kernel_sizes_) {
    feats.push_back(torch::max_pool2d(x, {ks, ks}, {1, 1}, {ks / 2, ks / 2}));
  }
  return conv2->forward(torch::cat(feats, 1));
}

// ---------------------------------------------------------------- CSPLayer
CSPLayerImpl::CSPLayerImpl(int64_t in_channels, int64_t out_channels,
                           int64_t n, bool shortcut, double expansion,
                           bool depthwise, Act act) {
  const auto hidden = static_cast<int64_t>(out_channels * expansion);
  conv1 = register_module("conv1",
                          BaseConv(in_channels, hidden, 1, 1, 1, false, act));
  conv2 = register_module("conv2",
                          BaseConv(in_channels, hidden, 1, 1, 1, false, act));
  conv3 = register_module(
      "conv3", BaseConv(2 * hidden, out_channels, 1, 1, 1, false, act));
  torch::nn::Sequential seq;
  for (int64_t i = 0; i < n; ++i) {
    seq->push_back(Bottleneck(hidden, hidden, shortcut, 1.0, depthwise, act));
  }
  m = register_module("m", seq);
}

torch::Tensor CSPLayerImpl::forward(const torch::Tensor& x) {
  auto x_1 = m->forward(conv1->forward(x));
  auto x_2 = conv2->forward(x);
  return conv3->forward(torch::cat({x_1, x_2}, 1));
}

// ---------------------------------------------------------------- Focus
FocusImpl::FocusImpl(int64_t in_channels, int64_t out_channels, int64_t ksize,
                     int64_t stride, Act act) {
  conv = register_module("conv", BaseConv(in_channels * 4, out_channels, ksize,
                                          stride, 1, false, act));
}

torch::Tensor FocusImpl::forward(const torch::Tensor& x) {
  auto tl = x.index({"...", Slice(None, None, 2), Slice(None, None, 2)});
  auto tr = x.index({"...", Slice(None, None, 2), Slice(1, None, 2)});
  auto bl = x.index({"...", Slice(1, None, 2), Slice(None, None, 2)});
  auto br = x.index({"...", Slice(1, None, 2), Slice(1, None, 2)});
  // NB: Python order is (top_left, bot_left, top_right, bot_right)
  return conv->forward(torch::cat({tl, bl, tr, br}, 1));
}

}  // namespace yolox
