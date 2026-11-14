// C++/LibTorch port of yolox/utils/boxes.py
#pragma once

#include <torch/torch.h>

#include <vector>

namespace yolox {

// IoU between every box in a [N,4] and b [M,4] -> [N,M]
torch::Tensor bboxes_iou(const torch::Tensor& a, const torch::Tensor& b,
                         bool xyxy = true);

// Greedy NMS (torchvision.ops.nms equivalent). Boxes xyxy [N,4], scores [N].
// Returns kept indices sorted by decreasing score. Runs on CPU.
torch::Tensor nms(const torch::Tensor& boxes, const torch::Tensor& scores,
                  double iou_threshold);

// torchvision.ops.batched_nms equivalent (coordinate-offset trick).
torch::Tensor batched_nms(const torch::Tensor& boxes, const torch::Tensor& scores,
                          const torch::Tensor& idxs, double iou_threshold);

// prediction: [B, N, 5 + num_classes] decoded (cx, cy, w, h, obj, cls...)
// Returns per image either an undefined tensor (no detections) or [K, 7]:
// (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
std::vector<torch::Tensor> postprocess(torch::Tensor prediction,
                                       int64_t num_classes, double conf_thre,
                                       double nms_thre,
                                       bool class_agnostic = false);

}  // namespace yolox
