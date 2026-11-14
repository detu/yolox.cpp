// C++/LibTorch port of yolox/models/yolo_head.py and yolox/models/losses.py
// (decoupled head, SimOTA label assignment, IoU/BCE/L1 losses).
#pragma once

#include <array>
#include <tuple>

#include "models/network_blocks.hpp"

namespace yolox {

struct LossOutput {
  torch::Tensor total_loss;
  torch::Tensor iou_loss;   // already multiplied by reg_weight (5.0)
  torch::Tensor conf_loss;
  torch::Tensor cls_loss;
  torch::Tensor l1_loss;    // zero tensor unless use_l1
  double num_fg = 0.0;      // num_fg / num_gts
};

struct YOLOXHeadImpl : torch::nn::Module {
  YOLOXHeadImpl(int64_t num_classes, double width = 1.0,
                std::array<int64_t, 3> strides = {8, 16, 32},
                std::array<int64_t, 3> in_channels = {256, 512, 1024},
                Act act = Act::SiLU, bool depthwise = false);

  void initialize_biases(double prior_prob);

  // Inference: returns [B, N_anchors, 5 + num_classes] decoded boxes
  // (cx, cy, w, h, obj_sigmoid, cls_sigmoid...). decode_in_inference=false
  // returns raw grid-relative outputs instead.
  torch::Tensor forward(const std::vector<torch::Tensor>& xin);

  // Training: labels [B, max_labels, 5] = (cls, cx, cy, w, h), zero padded.
  LossOutput forward_train(const std::vector<torch::Tensor>& xin,
                           const torch::Tensor& labels);

  int64_t num_classes;
  bool decode_in_inference = true;
  bool use_l1 = false;
  std::array<int64_t, 3> strides;

  torch::nn::ModuleList cls_convs{nullptr}, reg_convs{nullptr},
      cls_preds{nullptr}, reg_preds{nullptr}, obj_preds{nullptr},
      stems{nullptr};

 private:
  // typed handles to the modules registered in the ModuleLists above
  std::vector<BaseConv> stems_;
  std::vector<torch::nn::Sequential> cls_convs_, reg_convs_;
  std::vector<torch::nn::Conv2d> cls_preds_, reg_preds_, obj_preds_;

  struct LevelOut {
    torch::Tensor reg, obj, cls;
  };
  LevelOut run_level(size_t k, const torch::Tensor& x);

  std::pair<torch::Tensor, torch::Tensor> get_output_and_grid(
      const torch::Tensor& output, int64_t stride);

  torch::Tensor decode_outputs(const torch::Tensor& outputs,
                               const std::vector<std::pair<int64_t, int64_t>>& hw);

  LossOutput get_losses(const torch::Tensor& x_shifts,
                        const torch::Tensor& y_shifts,
                        const torch::Tensor& expanded_strides,
                        const torch::Tensor& labels,
                        const torch::Tensor& outputs,
                        const torch::Tensor& origin_preds);

  struct Assignment {
    torch::Tensor gt_matched_classes;
    torch::Tensor fg_mask;
    torch::Tensor pred_ious_this_matching;
    torch::Tensor matched_gt_inds;
    int64_t num_fg = 0;
  };

  Assignment get_assignments(int64_t batch_idx, int64_t num_gt,
                             const torch::Tensor& gt_bboxes_per_image,
                             const torch::Tensor& gt_classes,
                             const torch::Tensor& bboxes_preds_per_image,
                             const torch::Tensor& expanded_strides,
                             const torch::Tensor& x_shifts,
                             const torch::Tensor& y_shifts,
                             const torch::Tensor& cls_preds,
                             const torch::Tensor& obj_preds);

  std::pair<torch::Tensor, torch::Tensor> get_geometry_constraint(
      const torch::Tensor& gt_bboxes_per_image,
      const torch::Tensor& expanded_strides, const torch::Tensor& x_shifts,
      const torch::Tensor& y_shifts);

  void simota_matching(const torch::Tensor& cost,
                       const torch::Tensor& pair_wise_ious,
                       const torch::Tensor& gt_classes, int64_t num_gt,
                       torch::Tensor& fg_mask, Assignment& out);
};
TORCH_MODULE(YOLOXHead);

// IOUloss(reduction="none", loss_type="iou"|"giou"); boxes are (cx, cy, w, h)
torch::Tensor iou_loss(const torch::Tensor& pred, const torch::Tensor& target,
                       bool giou = false);

}  // namespace yolox
