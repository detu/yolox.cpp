// Mixed precision helpers. LibTorch has CUDA autocast but no GradScaler in
// C++, so a small dynamic loss scaler equivalent to torch.cuda.amp.GradScaler
// (init_scale=2**16, growth_factor=2, backoff_factor=0.5, growth_interval=2000)
// is implemented here.
#pragma once

#include <ATen/autocast_mode.h>
#include <torch/torch.h>
#include <torch/version.h>

namespace yolox {

inline void set_cuda_autocast(bool enabled) {
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
  at::autocast::set_autocast_enabled(at::kCUDA, enabled);
#else
  at::autocast::set_enabled(enabled);
#endif
}

// RAII: `with torch.cuda.amp.autocast(enabled=...)`
class AutocastGuard {
 public:
  explicit AutocastGuard(bool enabled) : enabled_(enabled) {
    if (enabled_) {
      set_cuda_autocast(true);
      at::autocast::increment_nesting();
    }
  }
  ~AutocastGuard() {
    if (enabled_) {
      if (at::autocast::decrement_nesting() == 0) at::autocast::clear_cache();
      set_cuda_autocast(false);
    }
  }
  AutocastGuard(const AutocastGuard&) = delete;
  AutocastGuard& operator=(const AutocastGuard&) = delete;

 private:
  bool enabled_;
};

class GradScaler {
 public:
  explicit GradScaler(bool enabled) : enabled_(enabled) {}

  torch::Tensor scale(const torch::Tensor& loss) const {
    return enabled_ ? loss * scale_ : loss;
  }

  // unscale grads, skip the step if any grad is inf/nan, update the scale
  template <typename Optimizer>
  bool step(Optimizer& opt, const std::vector<torch::Tensor>& params) {
    if (!enabled_) {
      opt.step();
      return true;
    }
    torch::NoGradGuard ng;
    const double inv = 1.0 / scale_;
    torch::Tensor found_inf;
    for (const auto& p : params) {
      auto& g = p.grad();
      if (!g.defined()) continue;
      g.mul_(inv);
      auto bad = g.isfinite().logical_not().any();
      found_inf = found_inf.defined() ? found_inf.logical_or(bad) : bad;
    }
    const bool skip = found_inf.defined() && found_inf.item<bool>();
    if (!skip) opt.step();
    update(skip);
    return !skip;
  }

  double get_scale() const { return scale_; }

 private:
  void update(bool found_inf) {
    if (found_inf) {
      scale_ *= 0.5;
      growth_tracker_ = 0;
    } else if (++growth_tracker_ == 2000) {
      scale_ *= 2.0;
      growth_tracker_ = 0;
    }
  }
  bool enabled_;
  double scale_ = 65536.0;
  int growth_tracker_ = 0;
};

}  // namespace yolox
