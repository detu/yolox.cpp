// C++/LibTorch port of yolox/models/darknet.py (CSPDarknet) and
// yolox/models/yolo_pafpn.py (YOLOPAFPN).
#pragma once

#include <array>

#include "models/network_blocks.hpp"

namespace yolox {

struct CSPDarknetImpl : torch::nn::Module {
  CSPDarknetImpl(double dep_mul, double wid_mul, bool depthwise = false,
                 Act act = Act::SiLU);
  // returns (dark3, dark4, dark5)
  std::array<torch::Tensor, 3> forward(const torch::Tensor& x);

  Focus stem{nullptr};
  torch::nn::Sequential dark2{nullptr}, dark3{nullptr}, dark4{nullptr},
      dark5{nullptr};
};
TORCH_MODULE(CSPDarknet);

struct YOLOPAFPNImpl : torch::nn::Module {
  YOLOPAFPNImpl(double depth = 1.0, double width = 1.0,
                std::array<int64_t, 3> in_channels = {256, 512, 1024},
                bool depthwise = false, Act act = Act::SiLU);
  // returns (pan_out2, pan_out1, pan_out0) at strides 8, 16, 32
  std::vector<torch::Tensor> forward(const torch::Tensor& x);

  CSPDarknet backbone{nullptr};
  BaseConv lateral_conv0{nullptr};
  CSPLayer C3_p4{nullptr};
  BaseConv reduce_conv1{nullptr};
  CSPLayer C3_p3{nullptr};
  torch::nn::AnyModule bu_conv2;
  CSPLayer C3_n3{nullptr};
  torch::nn::AnyModule bu_conv1;
  CSPLayer C3_n4{nullptr};
};
TORCH_MODULE(YOLOPAFPN);

}  // namespace yolox
