// C++ port of yolox/utils/lr_scheduler.py (cos, warmcos, yoloxwarmcos)
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace yolox {

constexpr double kPi = 3.14159265358979323846;

class LRScheduler {
 public:
  LRScheduler(std::string name, double lr, int64_t iters_per_epoch, int64_t total_epochs,
              int64_t warmup_epochs, double warmup_lr_start, int64_t no_aug_epochs,
              double min_lr_ratio)
      : name_(std::move(name)),
        lr_(lr),
        total_iters_(iters_per_epoch * total_epochs),
        warmup_total_iters_(iters_per_epoch * warmup_epochs),
        no_aug_iters_(iters_per_epoch * no_aug_epochs),
        warmup_lr_start_(warmup_lr_start),
        min_lr_ratio_(min_lr_ratio) {
    if (name_ != "cos" && name_ != "warmcos" && name_ != "yoloxwarmcos")
      throw std::invalid_argument("Scheduler version " + name_ + " not supported.");
  }

  double update_lr(int64_t iters) const {
    const double it = static_cast<double>(iters);
    if (name_ == "cos") {
      return lr_ * 0.5 * (1.0 + std::cos(kPi * it / total_iters_));
    }
    if (name_ == "warmcos") {
      if (iters <= warmup_total_iters_)
        return (lr_ - warmup_lr_start_) * it / static_cast<double>(warmup_total_iters_) +
               warmup_lr_start_;
      return lr_ * 0.5 *
             (1.0 + std::cos(kPi * (it - warmup_total_iters_) /
                             static_cast<double>(total_iters_ - warmup_total_iters_)));
    }
    // yoloxwarmcos
    const double min_lr = lr_ * min_lr_ratio_;
    if (iters <= warmup_total_iters_) {
      return (lr_ - warmup_lr_start_) *
                 std::pow(it / static_cast<double>(warmup_total_iters_), 2) +
             warmup_lr_start_;
    }
    if (iters >= total_iters_ - no_aug_iters_) return min_lr;
    return min_lr + 0.5 * (lr_ - min_lr) *
                        (1.0 + std::cos(kPi * (it - warmup_total_iters_) /
                                        static_cast<double>(total_iters_ - warmup_total_iters_ -
                                                            no_aug_iters_)));
  }

 private:
  std::string name_;
  double lr_;
  int64_t total_iters_, warmup_total_iters_, no_aug_iters_;
  double warmup_lr_start_, min_lr_ratio_;
};

}  // namespace yolox
