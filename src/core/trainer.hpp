// C++ port of yolox/core/trainer.py (single GPU)
#pragma once

#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "core/amp.hpp"
#include "core/ema.hpp"
#include "core/evaluator.hpp"
#include "core/lr_scheduler.hpp"
#include "data/dataloader.hpp"
#include "exp/exp.hpp"

namespace yolox {

struct TrainArgs {
  std::string experiment_name;
  int batch_size = 64;
  std::string ckpt;                 // -c
  bool resume = false;              // --resume
  std::optional<int> start_epoch;   // -e
  bool fp16 = false;                // --fp16
  bool cache = false;               // --cache (RAM)
  int device = 0;                   // CUDA device index, -1 = CPU
};

class Trainer {
 public:
  Trainer(Exp exp, TrainArgs args);
  void train();

 private:
  void before_train();
  void train_in_epoch();
  void train_one_iter();
  void before_epoch();
  void after_epoch();
  void after_iter();
  void after_train();
  void resume_train();
  void evaluate_and_save_model();
  void save_ckpt(const std::string& name, bool update_best = false,
                 std::optional<double> ap = std::nullopt);
  std::pair<int, int> random_resize();
  void preprocess(torch::Tensor& inputs, torch::Tensor& targets, std::pair<int, int> tsize) const;
  void build_optimizer();
  int64_t progress_in_iter() const { return epoch_ * max_iter_ + iter_; }
  void log(const std::string& msg);

  Exp exp_;
  TrainArgs args_;
  torch::Device device_{torch::kCPU};
  std::string file_name_;
  std::ofstream log_file_;

  YOLOX model_{nullptr};
  std::unique_ptr<torch::optim::SGD> optimizer_;
  std::vector<torch::Tensor> params_;
  std::unique_ptr<LRScheduler> lr_scheduler_;
  std::unique_ptr<ModelEMA> ema_model_;
  std::unique_ptr<TrainLoader> train_loader_;
  std::unique_ptr<COCOEvaluator> evaluator_;
  GradScaler scaler_{false};
  Rng rng_;

  int max_epoch_ = 0;
  int start_epoch_ = 0;
  int epoch_ = 0;
  int64_t iter_ = 0;
  int64_t max_iter_ = 0;
  bool no_aug_ = false;
  double best_ap_ = 0.0;
  std::pair<int, int> input_size_;  // (h, w), changes with multiscale training

  // MeterBuffer (window = print_interval)
  struct Meter {
    std::vector<double> window;
    double total = 0;
    int64_t count = 0;
    double latest() const { return window.empty() ? 0 : window.back(); }
    double avg() const {
      double s = 0;
      for (double v : window) s += v;
      return window.empty() ? 0 : s / window.size();
    }
    double global_avg() const { return count ? total / count : 0; }
    void update(double v, size_t win) {
      window.push_back(v);
      if (window.size() > win) window.erase(window.begin());
      total += v;
      ++count;
    }
    void clear() {  // AverageMeter.clear() also resets the global stats
      window.clear();
      total = 0;
      count = 0;
    }
  };
  Meter m_iter_time_, m_data_time_, m_lr_, m_total_, m_iou_, m_l1_, m_conf_, m_cls_, m_numfg_;
};

}  // namespace yolox
