#include "models/yolo_head.hpp"

#include <cmath>

#include "utils/boxes.hpp"

namespace yolox {

using torch::indexing::Slice;

// ---------------------------------------------------------------- IOUloss
torch::Tensor iou_loss(const torch::Tensor& pred_in, const torch::Tensor& target_in,
                       bool giou) {
  TORCH_CHECK(pred_in.size(0) == target_in.size(0));
  auto pred = pred_in.reshape({-1, 4});
  auto target = target_in.reshape({-1, 4});
  auto p_xy = pred.slice(1, 0, 2), p_wh = pred.slice(1, 2, 4);
  auto t_xy = target.slice(1, 0, 2), t_wh = target.slice(1, 2, 4);

  auto tl = torch::max(p_xy - p_wh / 2, t_xy - t_wh / 2);
  auto br = torch::min(p_xy + p_wh / 2, t_xy + t_wh / 2);
  auto area_p = torch::prod(p_wh, 1);
  auto area_g = torch::prod(t_wh, 1);
  auto en = (tl < br).to(tl.scalar_type()).prod(1);
  auto area_i = torch::prod(br - tl, 1) * en;
  auto area_u = area_p + area_g - area_i;
  auto iou = area_i / (area_u + 1e-16);

  if (!giou) return 1 - iou.pow(2);

  auto c_tl = torch::min(p_xy - p_wh / 2, t_xy - t_wh / 2);
  auto c_br = torch::max(p_xy + p_wh / 2, t_xy + t_wh / 2);
  auto area_c = torch::prod(c_br - c_tl, 1);
  auto g = iou - (area_c - area_u) / area_c.clamp_min(1e-16);
  return 1 - g.clamp(-1.0, 1.0);
}

// ---------------------------------------------------------------- YOLOXHead
YOLOXHeadImpl::YOLOXHeadImpl(int64_t num_classes_, double width,
                             std::array<int64_t, 3> strides_,
                             std::array<int64_t, 3> in_channels, Act act,
                             bool depthwise)
    : num_classes(num_classes_), strides(strides_) {
  cls_convs = register_module("cls_convs", torch::nn::ModuleList());
  reg_convs = register_module("reg_convs", torch::nn::ModuleList());
  cls_preds = register_module("cls_preds", torch::nn::ModuleList());
  reg_preds = register_module("reg_preds", torch::nn::ModuleList());
  obj_preds = register_module("obj_preds", torch::nn::ModuleList());
  stems = register_module("stems", torch::nn::ModuleList());

  const auto hidden = static_cast<int64_t>(256 * width);
  for (size_t i = 0; i < in_channels.size(); ++i) {
    BaseConv stem(static_cast<int64_t>(in_channels[i] * width), hidden, 1, 1, 1,
                  false, act);
    stems->push_back(stem);
    stems_.push_back(stem);

    torch::nn::Sequential cls_seq, reg_seq;
    cls_seq->push_back(make_conv(depthwise, hidden, hidden, 3, 1, act));
    cls_seq->push_back(make_conv(depthwise, hidden, hidden, 3, 1, act));
    reg_seq->push_back(make_conv(depthwise, hidden, hidden, 3, 1, act));
    reg_seq->push_back(make_conv(depthwise, hidden, hidden, 3, 1, act));
    cls_convs->push_back(cls_seq);
    reg_convs->push_back(reg_seq);
    cls_convs_.push_back(cls_seq);
    reg_convs_.push_back(reg_seq);

    auto mk = [hidden](int64_t out) {
      return torch::nn::Conv2d(
          torch::nn::Conv2dOptions(hidden, out, 1).stride(1).padding(0).bias(true));
    };
    auto cp = mk(num_classes), rp = mk(4), op = mk(1);
    cls_preds->push_back(cp);
    reg_preds->push_back(rp);
    obj_preds->push_back(op);
    cls_preds_.push_back(cp);
    reg_preds_.push_back(rp);
    obj_preds_.push_back(op);
  }
}

void YOLOXHeadImpl::initialize_biases(double prior_prob) {
  torch::NoGradGuard ng;
  const double b = -std::log((1 - prior_prob) / prior_prob);
  for (auto& c : cls_preds_) c->bias.fill_(b);
  for (auto& c : obj_preds_) c->bias.fill_(b);
}

YOLOXHeadImpl::LevelOut YOLOXHeadImpl::run_level(size_t k, const torch::Tensor& xin) {
  auto x = stems_[k]->forward(xin);
  auto cls_feat = cls_convs_[k]->forward(x);
  auto reg_feat = reg_convs_[k]->forward(x);
  return {reg_preds_[k]->forward(reg_feat), obj_preds_[k]->forward(reg_feat),
          cls_preds_[k]->forward(cls_feat)};
}

static torch::Tensor make_grid(int64_t h, int64_t w, const torch::TensorOptions& opts) {
  auto yx = torch::meshgrid({torch::arange(h, opts), torch::arange(w, opts)}, "ij");
  return torch::stack({yx[1], yx[0]}, 2).view({1, h * w, 2});  // (x, y)
}

std::pair<torch::Tensor, torch::Tensor> YOLOXHeadImpl::get_output_and_grid(
    const torch::Tensor& output_in, int64_t stride) {
  const int64_t B = output_in.size(0);
  const int64_t n_ch = 5 + num_classes;
  const int64_t h = output_in.size(2), w = output_in.size(3);
  auto grid = make_grid(h, w, output_in.options());

  auto output = output_in.view({B, 1, n_ch, h, w})
                    .permute({0, 1, 3, 4, 2})
                    .reshape({B, h * w, -1});
  auto xy = (output.slice(2, 0, 2) + grid) * static_cast<double>(stride);
  auto wh = torch::exp(output.slice(2, 2, 4)) * static_cast<double>(stride);
  output = torch::cat({xy, wh, output.slice(2, 4)}, 2);
  return {output, grid};
}

torch::Tensor YOLOXHeadImpl::decode_outputs(
    const torch::Tensor& outputs, const std::vector<std::pair<int64_t, int64_t>>& hw) {
  std::vector<torch::Tensor> grids, strides_v;
  auto opts = outputs.options();
  for (size_t i = 0; i < hw.size(); ++i) {
    auto g = make_grid(hw[i].first, hw[i].second, opts);
    grids.push_back(g);
    strides_v.push_back(torch::full({1, g.size(1), 1}, static_cast<double>(strides[i]), opts));
  }
  auto grid = torch::cat(grids, 1);
  auto st = torch::cat(strides_v, 1);
  return torch::cat({(outputs.slice(2, 0, 2) + grid) * st,
                     torch::exp(outputs.slice(2, 2, 4)) * st, outputs.slice(2, 4)},
                    -1);
}

torch::Tensor YOLOXHeadImpl::forward(const std::vector<torch::Tensor>& xin) {
  std::vector<torch::Tensor> outputs;
  std::vector<std::pair<int64_t, int64_t>> hw;
  for (size_t k = 0; k < xin.size(); ++k) {
    auto lo = run_level(k, xin[k]);
    auto out = torch::cat({lo.reg, lo.obj.sigmoid(), lo.cls.sigmoid()}, 1);
    hw.emplace_back(out.size(2), out.size(3));
    outputs.push_back(out.flatten(2));
  }
  auto out = torch::cat(outputs, 2).permute({0, 2, 1});
  if (decode_in_inference) return decode_outputs(out, hw);
  return out;
}

LossOutput YOLOXHeadImpl::forward_train(const std::vector<torch::Tensor>& xin,
                                        const torch::Tensor& labels) {
  std::vector<torch::Tensor> outputs, origin_preds, x_shifts, y_shifts, exp_strides;
  for (size_t k = 0; k < xin.size(); ++k) {
    auto lo = run_level(k, xin[k]);
    auto output = torch::cat({lo.reg, lo.obj, lo.cls}, 1);
    auto og = get_output_and_grid(output, strides[k]);
    auto& grid = og.second;
    x_shifts.push_back(grid.select(2, 0));  // [1, HW]
    y_shifts.push_back(grid.select(2, 1));
    exp_strides.push_back(torch::full({1, grid.size(1)}, static_cast<double>(strides[k]),
                                      xin[0].options()));
    if (use_l1) {
      const int64_t B = lo.reg.size(0), h = lo.reg.size(2), w = lo.reg.size(3);
      origin_preds.push_back(lo.reg.view({B, 1, 4, h, w})
                                 .permute({0, 1, 3, 4, 2})
                                 .reshape({B, -1, 4})
                                 .clone());
    }
    outputs.push_back(og.first);
  }
  torch::Tensor origin = use_l1 ? torch::cat(origin_preds, 1) : torch::Tensor();
  return get_losses(torch::cat(x_shifts, 1), torch::cat(y_shifts, 1),
                    torch::cat(exp_strides, 1), labels, torch::cat(outputs, 1), origin);
}

LossOutput YOLOXHeadImpl::get_losses(const torch::Tensor& x_shifts,
                                     const torch::Tensor& y_shifts,
                                     const torch::Tensor& expanded_strides,
                                     const torch::Tensor& labels,
                                     const torch::Tensor& outputs,
                                     const torch::Tensor& origin_preds) {
  auto bbox_preds = outputs.slice(2, 0, 4);  // [B, A, 4]
  auto obj_preds = outputs.slice(2, 4, 5);   // [B, A, 1]
  auto cls_preds = outputs.slice(2, 5);      // [B, A, C]

  auto nlabel = (labels.sum(2) > 0).sum(1).to(torch::kCPU);
  auto nlabel_a = nlabel.accessor<int64_t, 1>();
  const int64_t total_num_anchors = outputs.size(1);
  const int64_t B = outputs.size(0);
  auto fopts = outputs.options().dtype(torch::kFloat);

  std::vector<torch::Tensor> cls_targets, reg_targets, l1_targets, obj_targets, fg_masks;
  double num_fg = 0.0, num_gts = 0.0;

  for (int64_t b = 0; b < B; ++b) {
    int64_t num_gt = nlabel_a[b];
    num_gts += static_cast<double>(num_gt);
    torch::Tensor cls_t, reg_t, l1_t, obj_t, fg_mask;

    Assignment asg;
    torch::Tensor gt_bboxes, gt_classes;
    if (num_gt > 0) {
      gt_bboxes = labels[b].slice(0, 0, num_gt).slice(1, 1, 5);
      gt_classes = labels[b].slice(0, 0, num_gt).select(1, 0);
      asg = get_assignments(b, num_gt, gt_bboxes, gt_classes, bbox_preds[b],
                            expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds);
      if (!asg.fg_mask.defined()) num_gt = 0;  // no candidate anchors at all
    }

    if (num_gt == 0) {
      cls_t = torch::zeros({0, num_classes}, fopts);
      reg_t = torch::zeros({0, 4}, fopts);
      l1_t = torch::zeros({0, 4}, fopts);
      obj_t = torch::zeros({total_num_anchors, 1}, fopts);
      fg_mask = torch::zeros({total_num_anchors}, fopts.dtype(torch::kBool));
    } else {
      fg_mask = asg.fg_mask;
      num_fg += static_cast<double>(asg.num_fg);
      cls_t = torch::one_hot(asg.gt_matched_classes.to(torch::kLong), num_classes)
                  .to(torch::kFloat) *
              asg.pred_ious_this_matching.unsqueeze(-1);
      obj_t = fg_mask.unsqueeze(-1).to(torch::kFloat);
      reg_t = gt_bboxes.index_select(0, asg.matched_gt_inds).to(torch::kFloat);
      if (use_l1) {
        auto gt = reg_t;
        auto st = expanded_strides[0].index({fg_mask}).to(torch::kFloat);
        auto xs = x_shifts[0].index({fg_mask}).to(torch::kFloat);
        auto ys = y_shifts[0].index({fg_mask}).to(torch::kFloat);
        const double eps = 1e-8;
        l1_t = torch::stack({gt.select(1, 0) / st - xs, gt.select(1, 1) / st - ys,
                             torch::log(gt.select(1, 2) / st + eps),
                             torch::log(gt.select(1, 3) / st + eps)},
                            1);
      }
    }
    cls_targets.push_back(cls_t);
    reg_targets.push_back(reg_t);
    obj_targets.push_back(obj_t);
    fg_masks.push_back(fg_mask);
    if (use_l1) l1_targets.push_back(l1_t.defined() ? l1_t : torch::zeros({0, 4}, fopts));
  }

  auto cls_targets_c = torch::cat(cls_targets, 0);
  auto reg_targets_c = torch::cat(reg_targets, 0);
  auto obj_targets_c = torch::cat(obj_targets, 0);
  auto fg_masks_c = torch::cat(fg_masks, 0);

  num_fg = std::max(num_fg, 1.0);
  auto none = at::Reduction::None;

  auto loss_iou =
      iou_loss(bbox_preds.reshape({-1, 4}).index({fg_masks_c}), reg_targets_c).sum() / num_fg;
  auto loss_obj = torch::binary_cross_entropy_with_logits(obj_preds.reshape({-1, 1}),
                                                          obj_targets_c, {}, {}, none)
                      .sum() /
                  num_fg;
  auto loss_cls =
      torch::binary_cross_entropy_with_logits(
          cls_preds.reshape({-1, num_classes}).index({fg_masks_c}), cls_targets_c, {}, {}, none)
          .sum() /
      num_fg;
  torch::Tensor loss_l1;
  if (use_l1) {
    auto l1_targets_c = torch::cat(l1_targets, 0);
    loss_l1 = (origin_preds.reshape({-1, 4}).index({fg_masks_c}).to(torch::kFloat) -
               l1_targets_c)
                  .abs()
                  .sum() /
              num_fg;
  } else {
    loss_l1 = torch::zeros({}, fopts);
  }

  const double reg_weight = 5.0;
  LossOutput out;
  out.total_loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1;
  out.iou_loss = reg_weight * loss_iou;
  out.conf_loss = loss_obj;
  out.cls_loss = loss_cls;
  out.l1_loss = loss_l1;
  out.num_fg = num_fg / std::max(num_gts, 1.0);
  return out;
}

std::pair<torch::Tensor, torch::Tensor> YOLOXHeadImpl::get_geometry_constraint(
    const torch::Tensor& gt, const torch::Tensor& expanded_strides,
    const torch::Tensor& x_shifts, const torch::Tensor& y_shifts) {
  auto es = expanded_strides[0];
  auto x_c = ((x_shifts[0] + 0.5) * es).unsqueeze(0);
  auto y_c = ((y_shifts[0] + 0.5) * es).unsqueeze(0);
  const double center_radius = 1.5;
  auto center_dist = es.unsqueeze(0) * center_radius;

  auto gl = gt.slice(1, 0, 1) - center_dist;
  auto gr = gt.slice(1, 0, 1) + center_dist;
  auto gtp = gt.slice(1, 1, 2) - center_dist;
  auto gb = gt.slice(1, 1, 2) + center_dist;

  auto c_l = x_c - gl;
  auto c_r = gr - x_c;
  auto c_t = y_c - gtp;
  auto c_b = gb - y_c;
  auto center_deltas = torch::stack({c_l, c_t, c_r, c_b}, 2);
  auto is_in_centers = std::get<0>(center_deltas.min(-1)) > 0.0;  // [G, A]
  auto anchor_filter = is_in_centers.sum(0) > 0;                  // [A]
  auto geometry_relation = is_in_centers.index({Slice(), anchor_filter});
  return {anchor_filter, geometry_relation};
}

YOLOXHeadImpl::Assignment YOLOXHeadImpl::get_assignments(
    int64_t batch_idx, int64_t num_gt, const torch::Tensor& gt_bboxes_in,
    const torch::Tensor& gt_classes_in, const torch::Tensor& bboxes_preds_in,
    const torch::Tensor& expanded_strides_in, const torch::Tensor& x_shifts_in,
    const torch::Tensor& y_shifts_in, const torch::Tensor& cls_preds,
    const torch::Tensor& obj_preds) {
  torch::NoGradGuard no_grad;
  // Label assignment is always done in fp32 (Python disables autocast here).
  auto gt_bboxes = gt_bboxes_in.to(torch::kFloat);
  auto gt_classes = gt_classes_in.to(torch::kFloat);
  auto expanded_strides = expanded_strides_in.to(torch::kFloat);
  auto x_shifts = x_shifts_in.to(torch::kFloat);
  auto y_shifts = y_shifts_in.to(torch::kFloat);

  auto gc = get_geometry_constraint(gt_bboxes, expanded_strides, x_shifts, y_shifts);
  auto fg_mask = gc.first;
  auto geometry_relation = gc.second;

  auto bboxes_preds = bboxes_preds_in.index({fg_mask}).to(torch::kFloat);
  auto cls_p = cls_preds[batch_idx].index({fg_mask}).to(torch::kFloat);
  auto obj_p = obj_preds[batch_idx].index({fg_mask}).to(torch::kFloat);
  const int64_t num_in_boxes_anchor = bboxes_preds.size(0);

  Assignment out;
  if (num_in_boxes_anchor == 0) return out;  // fg_mask undefined -> caller treats as no gt

  auto pair_wise_ious = bboxes_iou(gt_bboxes, bboxes_preds, /*xyxy=*/false);
  auto pair_wise_ious_loss = -torch::log(pair_wise_ious + 1e-8);

  // BCE(sqrt(sig(cls)*sig(obj)), one_hot(gt)) summed over classes, for every
  // (gt, anchor) pair. Since the target is one-hot this equals
  //   -(log p[a, c_g] + sum_c log(1 - p[a, c]) - log(1 - p[a, c_g]))
  // which avoids materialising the [G, A, C] tensors. log is clamped at -100
  // exactly like torch's binary_cross_entropy.
  auto p = (cls_p.sigmoid() * obj_p.sigmoid()).sqrt();  // [A, C]
  auto logp = torch::log(p).clamp_min(-100.0);
  auto log1mp = torch::log(1 - p).clamp_min(-100.0);
  auto gidx = gt_classes.to(torch::kLong);
  auto sum_log1mp = log1mp.sum(1).unsqueeze(0);           // [1, A]
  auto logp_g = logp.index_select(1, gidx).t();           // [G, A]
  auto log1mp_g = log1mp.index_select(1, gidx).t();       // [G, A]
  auto pair_wise_cls_loss = -(logp_g + sum_log1mp - log1mp_g);

  auto cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss +
              1e6 * (~geometry_relation).to(torch::kFloat);

  simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask, out);
  return out;
}

void YOLOXHeadImpl::simota_matching(const torch::Tensor& cost,
                                    const torch::Tensor& pair_wise_ious,
                                    const torch::Tensor& gt_classes, int64_t num_gt,
                                    torch::Tensor& fg_mask, Assignment& out) {
  auto matching_matrix = torch::zeros_like(cost, cost.options().dtype(torch::kUInt8));
  const int64_t n_candidate_k = std::min<int64_t>(10, pair_wise_ious.size(1));
  auto topk_ious = std::get<0>(pair_wise_ious.topk(n_candidate_k, 1));
  auto dynamic_ks = topk_ious.sum(1).to(torch::kInt).clamp_min(1).to(torch::kCPU).to(torch::kLong);
  auto ks = dynamic_ks.accessor<int64_t, 1>();

  for (int64_t g = 0; g < num_gt; ++g) {
    auto pos_idx = std::get<1>(cost[g].topk(ks[g], 0, /*largest=*/false));
    matching_matrix[g].index_fill_(0, pos_idx, 1);
  }

  auto anchor_matching_gt = matching_matrix.sum(0);
  if (anchor_matching_gt.max().item<int64_t>() > 1) {
    auto multiple_match_mask = anchor_matching_gt > 1;
    auto cost_argmin = std::get<1>(cost.index({Slice(), multiple_match_mask}).min(0));
    auto mm_idx = multiple_match_mask.nonzero().squeeze(1);
    matching_matrix.index_put_({Slice(), multiple_match_mask}, 0);
    matching_matrix.index_put_({cost_argmin, mm_idx},
                               torch::ones({mm_idx.size(0)}, matching_matrix.options()));
  }
  auto fg_mask_inboxes = anchor_matching_gt > 0;
  out.num_fg = fg_mask_inboxes.sum().item<int64_t>();

  // fg_mask[fg_mask.clone()] = fg_mask_inboxes
  auto fg_idx = fg_mask.nonzero().squeeze(1);
  auto new_fg = torch::zeros_like(fg_mask);
  new_fg.index_put_({fg_idx.index({fg_mask_inboxes})}, true);
  out.fg_mask = new_fg;

  out.matched_gt_inds = matching_matrix.index({Slice(), fg_mask_inboxes}).argmax(0);
  out.gt_matched_classes = gt_classes.index_select(0, out.matched_gt_inds);
  out.pred_ious_this_matching =
      (matching_matrix.to(torch::kFloat) * pair_wise_ious).sum(0).index({fg_mask_inboxes});
}

}  // namespace yolox
