// C++ port of yolox/utils/ema.py (ModelEMA)
#pragma once

#include <cmath>
#include <cstdint>

#include "core/checkpoint.hpp"
#include "models/yolox.hpp"

namespace yolox {

class ModelEMA {
 public:
  ModelEMA(const YOLOX& model, double decay = 0.9999, int64_t updates_ = 0)
      : decay_(decay), updates(updates_) {
    // deepcopy(model): build an identical network and copy all tensors
    ema = YOLOX(model->cfg);
    ema->to(model->parameters().front().device());
    torch::NoGradGuard ng;
    auto src = state_dict(*model);
    auto dst = state_dict(*ema);
    for (size_t i = 0; i < dst.size(); ++i) dst[i].second.copy_(src[i].second);
    ema->eval();
    for (auto& p : ema->parameters()) p.set_requires_grad(false);
    ema_sd_ = state_dict(*ema);
  }

  void update(const YOLOX& model) {
    torch::NoGradGuard ng;
    ++updates;
    const double d = decay_ * (1.0 - std::exp(-static_cast<double>(updates) / 2000.0));
    auto msd = state_dict(*model);
    for (size_t i = 0; i < ema_sd_.size(); ++i) {
      auto& v = ema_sd_[i].second;
      if (!v.is_floating_point()) continue;
      v.mul_(d).add_(msd[i].second.detach(), 1.0 - d);
    }
  }

  YOLOX ema{nullptr};

 private:
  double decay_;

 public:
  int64_t updates;

 private:
  StateDict ema_sd_;
};

}  // namespace yolox
