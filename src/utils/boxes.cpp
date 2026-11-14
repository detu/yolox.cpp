#include "utils/boxes.hpp"

#include <algorithm>
#include <numeric>

namespace yolox {

using torch::indexing::Slice;

torch::Tensor bboxes_iou(const torch::Tensor& a, const torch::Tensor& b,
                         bool xyxy) {
  TORCH_CHECK(a.size(1) == 4 && b.size(1) == 4, "bboxes_iou expects [N,4]");
  torch::Tensor tl, br, area_a, area_b;
  auto a_u = a.unsqueeze(1);  // [N,1,4]
  if (xyxy) {
    tl = torch::max(a_u.slice(2, 0, 2), b.slice(1, 0, 2));
    br = torch::min(a_u.slice(2, 2, 4), b.slice(1, 2, 4));
    area_a = torch::prod(a.slice(1, 2, 4) - a.slice(1, 0, 2), 1);
    area_b = torch::prod(b.slice(1, 2, 4) - b.slice(1, 0, 2), 1);
  } else {
    tl = torch::max(a_u.slice(2, 0, 2) - a_u.slice(2, 2, 4) / 2,
                    b.slice(1, 0, 2) - b.slice(1, 2, 4) / 2);
    br = torch::min(a_u.slice(2, 0, 2) + a_u.slice(2, 2, 4) / 2,
                    b.slice(1, 0, 2) + b.slice(1, 2, 4) / 2);
    area_a = torch::prod(a.slice(1, 2, 4), 1);
    area_b = torch::prod(b.slice(1, 2, 4), 1);
  }
  auto en = (tl < br).to(tl.scalar_type()).prod(2);
  auto area_i = torch::prod(br - tl, 2) * en;
  return area_i / (area_a.unsqueeze(1) + area_b - area_i);
}

torch::Tensor nms(const torch::Tensor& boxes_in, const torch::Tensor& scores_in,
                  double iou_threshold) {
  const int64_t n = boxes_in.size(0);
  if (n == 0) return torch::empty({0}, torch::kLong);
  auto boxes = boxes_in.detach().to(torch::kCPU, torch::kFloat).contiguous();
  auto scores = scores_in.detach().to(torch::kCPU, torch::kFloat).contiguous();
  const float* bx = boxes.data_ptr<float>();
  const float* sc = scores.data_ptr<float>();

  std::vector<int64_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [sc](int64_t i, int64_t j) { return sc[i] > sc[j]; });

  std::vector<float> areas(n);
  for (int64_t i = 0; i < n; ++i) {
    areas[i] = (bx[4 * i + 2] - bx[4 * i]) * (bx[4 * i + 3] - bx[4 * i + 1]);
  }
  std::vector<char> suppressed(n, 0);
  std::vector<int64_t> keep;
  keep.reserve(n);
  for (int64_t oi = 0; oi < n; ++oi) {
    const int64_t i = order[oi];
    if (suppressed[i]) continue;
    keep.push_back(i);
    const float ix1 = bx[4 * i], iy1 = bx[4 * i + 1], ix2 = bx[4 * i + 2],
                iy2 = bx[4 * i + 3];
    for (int64_t oj = oi + 1; oj < n; ++oj) {
      const int64_t j = order[oj];
      if (suppressed[j]) continue;
      const float xx1 = std::max(ix1, bx[4 * j]);
      const float yy1 = std::max(iy1, bx[4 * j + 1]);
      const float xx2 = std::min(ix2, bx[4 * j + 2]);
      const float yy2 = std::min(iy2, bx[4 * j + 3]);
      const float w = std::max(0.f, xx2 - xx1);
      const float h = std::max(0.f, yy2 - yy1);
      const float inter = w * h;
      const float iou = inter / (areas[i] + areas[j] - inter);
      if (iou > iou_threshold) suppressed[j] = 1;
    }
  }
  return torch::tensor(keep, torch::kLong);
}

torch::Tensor batched_nms(const torch::Tensor& boxes, const torch::Tensor& scores,
                          const torch::Tensor& idxs, double iou_threshold) {
  if (boxes.numel() == 0) return torch::empty({0}, torch::kLong);
  auto max_coordinate = boxes.max();
  auto offsets = idxs.to(boxes.scalar_type()) * (max_coordinate + 1);
  auto boxes_for_nms = boxes + offsets.unsqueeze(1);
  return nms(boxes_for_nms, scores, iou_threshold);
}

std::vector<torch::Tensor> postprocess(torch::Tensor prediction,
                                       int64_t num_classes, double conf_thre,
                                       double nms_thre, bool class_agnostic) {
  prediction = prediction.to(torch::kFloat);
  auto cx = prediction.select(2, 0), cy = prediction.select(2, 1);
  auto w = prediction.select(2, 2), h = prediction.select(2, 3);
  auto corners = torch::stack({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2}, 2);
  prediction = torch::cat({corners, prediction.slice(2, 4)}, 2);

  std::vector<torch::Tensor> output(prediction.size(0));
  for (int64_t i = 0; i < prediction.size(0); ++i) {
    auto image_pred = prediction[i];
    if (image_pred.size(0) == 0) continue;
    auto mx = image_pred.slice(1, 5, 5 + num_classes).max(1, /*keepdim=*/true);
    auto class_conf = std::get<0>(mx);
    auto class_pred = std::get<1>(mx);
    auto conf_mask = (image_pred.select(1, 4) * class_conf.squeeze(1)) >= conf_thre;
    auto detections =
        torch::cat({image_pred.slice(1, 0, 5), class_conf, class_pred.to(torch::kFloat)}, 1);
    detections = detections.index({conf_mask});
    if (detections.size(0) == 0) continue;

    auto det_scores = detections.select(1, 4) * detections.select(1, 5);
    torch::Tensor keep;
    if (class_agnostic) {
      keep = nms(detections.slice(1, 0, 4), det_scores, nms_thre);
    } else {
      keep = batched_nms(detections.slice(1, 0, 4), det_scores,
                         detections.select(1, 6), nms_thre);
    }
    output[i] = detections.index_select(0, keep.to(detections.device()));
  }
  return output;
}

}  // namespace yolox
