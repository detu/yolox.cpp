// C++/LibTorch port of yolox/models/network_blocks.py
//
// Module and member names are kept identical to the Python implementation so
// that state_dict keys match ("conv", "bn", "dconv", "pconv", "conv1", ...).
// That lets us load the official yolox_*.pth weights by name.
#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

namespace yolox {

enum class Act { SiLU, ReLU, LReLU };

Act act_from_string(const std::string& name);
torch::Tensor apply_act(const torch::Tensor& x, Act act);

// Conv2d -> BatchNorm2d -> activation
struct BaseConvImpl : torch::nn::Module {
  BaseConvImpl(int64_t in_channels, int64_t out_channels, int64_t ksize,
               int64_t stride, int64_t groups = 1, bool bias = false,
               Act act = Act::SiLU);
  torch::Tensor forward(const torch::Tensor& x);

  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  Act act_;
};
TORCH_MODULE(BaseConv);

// Depthwise conv + pointwise conv
struct DWConvImpl : torch::nn::Module {
  DWConvImpl(int64_t in_channels, int64_t out_channels, int64_t ksize,
             int64_t stride = 1, Act act = Act::SiLU);
  torch::Tensor forward(const torch::Tensor& x);

  BaseConv dconv{nullptr};
  BaseConv pconv{nullptr};
};
TORCH_MODULE(DWConv);

// Returns either a BaseConv or a DWConv wrapped in AnyModule
// (Python: `Conv = DWConv if depthwise else BaseConv`).
torch::nn::AnyModule make_conv(bool depthwise, int64_t in_channels,
                               int64_t out_channels, int64_t ksize,
                               int64_t stride, Act act);

struct BottleneckImpl : torch::nn::Module {
  BottleneckImpl(int64_t in_channels, int64_t out_channels, bool shortcut = true,
                 double expansion = 0.5, bool depthwise = false,
                 Act act = Act::SiLU);
  torch::Tensor forward(const torch::Tensor& x);

  BaseConv conv1{nullptr};
  torch::nn::AnyModule conv2;
  bool use_add;
};
TORCH_MODULE(Bottleneck);

// Spatial pyramid pooling (max pools have no params, so they are functional here)
struct SPPBottleneckImpl : torch::nn::Module {
  SPPBottleneckImpl(int64_t in_channels, int64_t out_channels,
                    std::vector<int64_t> kernel_sizes = {5, 9, 13},
                    Act act = Act::SiLU);
  torch::Tensor forward(const torch::Tensor& x);

  BaseConv conv1{nullptr};
  BaseConv conv2{nullptr};
  std::vector<int64_t> kernel_sizes_;
};
TORCH_MODULE(SPPBottleneck);

// C3 in yolov5, CSP bottleneck with 3 convolutions
struct CSPLayerImpl : torch::nn::Module {
  CSPLayerImpl(int64_t in_channels, int64_t out_channels, int64_t n = 1,
               bool shortcut = true, double expansion = 0.5,
               bool depthwise = false, Act act = Act::SiLU);
  torch::Tensor forward(const torch::Tensor& x);

  BaseConv conv1{nullptr};
  BaseConv conv2{nullptr};
  BaseConv conv3{nullptr};
  torch::nn::Sequential m{nullptr};
};
TORCH_MODULE(CSPLayer);

// Focus width and height information into channel space
struct FocusImpl : torch::nn::Module {
  FocusImpl(int64_t in_channels, int64_t out_channels, int64_t ksize = 1,
            int64_t stride = 1, Act act = Act::SiLU);
  torch::Tensor forward(const torch::Tensor& x);

  BaseConv conv{nullptr};
};
TORCH_MODULE(Focus);

}  // namespace yolox
