#include "models/yolo_pafpn.hpp"

#include <algorithm>
#include <cmath>

namespace yolox {

namespace {
// Python's round() is banker's rounding; for the depth multipliers YOLOX uses
// (0.33, 0.67, 1.0, 1.33) the products never land on .5, so nearbyint matches.
int64_t py_round(double v) { return static_cast<int64_t>(std::nearbyint(v)); }
}  // namespace

// ---------------------------------------------------------------- CSPDarknet
CSPDarknetImpl::CSPDarknetImpl(double dep_mul, double wid_mul, bool depthwise,
                               Act act) {
  const auto base_channels = static_cast<int64_t>(wid_mul * 64);
  const int64_t base_depth = std::max<int64_t>(py_round(dep_mul * 3), 1);

  stem = register_module("stem", Focus(3, base_channels, 3, 1, act));

  torch::nn::Sequential d2;
  d2->push_back(make_conv(depthwise, base_channels, base_channels * 2, 3, 2, act));
  d2->push_back(CSPLayer(base_channels * 2, base_channels * 2, base_depth, true,
                         0.5, depthwise, act));
  dark2 = register_module("dark2", d2);

  torch::nn::Sequential d3;
  d3->push_back(
      make_conv(depthwise, base_channels * 2, base_channels * 4, 3, 2, act));
  d3->push_back(CSPLayer(base_channels * 4, base_channels * 4, base_depth * 3,
                         true, 0.5, depthwise, act));
  dark3 = register_module("dark3", d3);

  torch::nn::Sequential d4;
  d4->push_back(
      make_conv(depthwise, base_channels * 4, base_channels * 8, 3, 2, act));
  d4->push_back(CSPLayer(base_channels * 8, base_channels * 8, base_depth * 3,
                         true, 0.5, depthwise, act));
  dark4 = register_module("dark4", d4);

  torch::nn::Sequential d5;
  d5->push_back(
      make_conv(depthwise, base_channels * 8, base_channels * 16, 3, 2, act));
  d5->push_back(SPPBottleneck(base_channels * 16, base_channels * 16,
                              {5, 9, 13}, act));
  d5->push_back(CSPLayer(base_channels * 16, base_channels * 16, base_depth,
                         /*shortcut=*/false, 0.5, depthwise, act));
  dark5 = register_module("dark5", d5);
}

std::array<torch::Tensor, 3> CSPDarknetImpl::forward(const torch::Tensor& input) {
  auto x = stem->forward(input);
  x = dark2->forward(x);
  auto d3 = dark3->forward(x);
  auto d4 = dark4->forward(d3);
  auto d5 = dark5->forward(d4);
  return {d3, d4, d5};
}

// ---------------------------------------------------------------- YOLOPAFPN
YOLOPAFPNImpl::YOLOPAFPNImpl(double depth, double width,
                             std::array<int64_t, 3> in_ch, bool depthwise,
                             Act act) {
  auto W = [width](int64_t c) { return static_cast<int64_t>(c * width); };
  const int64_t n = py_round(3 * depth);

  backbone = register_module("backbone",
                             CSPDarknet(depth, width, depthwise, act));
  lateral_conv0 = register_module(
      "lateral_conv0", BaseConv(W(in_ch[2]), W(in_ch[1]), 1, 1, 1, false, act));
  C3_p4 = register_module(
      "C3_p4", CSPLayer(static_cast<int64_t>(2 * in_ch[1] * width), W(in_ch[1]),
                        n, false, 0.5, depthwise, act));
  reduce_conv1 = register_module(
      "reduce_conv1", BaseConv(W(in_ch[1]), W(in_ch[0]), 1, 1, 1, false, act));
  C3_p3 = register_module(
      "C3_p3", CSPLayer(static_cast<int64_t>(2 * in_ch[0] * width), W(in_ch[0]),
                        n, false, 0.5, depthwise, act));
  bu_conv2 = make_conv(depthwise, W(in_ch[0]), W(in_ch[0]), 3, 2, act);
  register_module("bu_conv2", bu_conv2.ptr());
  C3_n3 = register_module(
      "C3_n3", CSPLayer(static_cast<int64_t>(2 * in_ch[0] * width), W(in_ch[1]),
                        n, false, 0.5, depthwise, act));
  bu_conv1 = make_conv(depthwise, W(in_ch[1]), W(in_ch[1]), 3, 2, act);
  register_module("bu_conv1", bu_conv1.ptr());
  C3_n4 = register_module(
      "C3_n4", CSPLayer(static_cast<int64_t>(2 * in_ch[1] * width), W(in_ch[2]),
                        n, false, 0.5, depthwise, act));
}

std::vector<torch::Tensor> YOLOPAFPNImpl::forward(const torch::Tensor& input) {
  namespace F = torch::nn::functional;
  auto up = [](const torch::Tensor& t) {
    return F::interpolate(t, F::InterpolateFuncOptions()
                                 .scale_factor(std::vector<double>{2.0, 2.0})
                                 .mode(torch::kNearest));
  };

  auto feats = backbone->forward(input);
  auto& x2 = feats[0];
  auto& x1 = feats[1];
  auto& x0 = feats[2];

  auto fpn_out0 = lateral_conv0->forward(x0);                     // 1024->512/32
  auto f_out0 = C3_p4->forward(torch::cat({up(fpn_out0), x1}, 1));  // 1024->512/16
  auto fpn_out1 = reduce_conv1->forward(f_out0);                  // 512->256/16
  auto pan_out2 = C3_p3->forward(torch::cat({up(fpn_out1), x2}, 1));  // 512->256/8

  auto p_out1 = bu_conv2.forward(pan_out2);                       // 256->256/16
  auto pan_out1 = C3_n3->forward(torch::cat({p_out1, fpn_out1}, 1));  // 512->512/16
  auto p_out0 = bu_conv1.forward(pan_out1);                       // 512->512/32
  auto pan_out0 = C3_n4->forward(torch::cat({p_out0, fpn_out0}, 1));  // 1024->1024/32

  return {pan_out2, pan_out1, pan_out0};
}

}  // namespace yolox
